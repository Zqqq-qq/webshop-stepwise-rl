"""纯CPU契约测试，不作为Qwen或GPU实训证据。"""
import copy
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
try:
    import torch
except ImportError:
    raise unittest.SkipTest('本组PRM测试需安装PyTorch；不作为GPU证据')
from torch import nn
from webshop_lab.prm.model_core import NeuralPRM, save_prm, restore_head, verify_prm
from webshop_lab.prm.train_core import accumulated_optimizer_step, optimizer_step


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=7)
        self.embedding = nn.Embedding(7, 5)
        self.projection = nn.Linear(5, 7)
    def forward(self, input_ids, attention_mask, use_cache=False):
        hidden = self.embedding(input_ids).cumsum(dim=1)
        return SimpleNamespace(logits=self.projection(hidden))
    def save_pretrained(self, path):
        path.mkdir()
        # 仅用于触发文件契约；真实PEFT入口须单独验证。
        (path/'adapter_model.safetensors').write_bytes('CPU契约占位'.encode('utf-8'))


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(7,5)

    def forward(self,input_ids,attention_mask,use_cache=False,return_dict=True):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids).cumsum(dim=1))


class TinyHiddenPeft(nn.Module):
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(vocab_size=7,hidden_size=5)
        self.decoder_base=nn.Module()
        self.decoder_base.model=TinyDecoder()

    def get_base_model(self):
        return self.decoder_base

    def forward(self,*args,**kwargs):
        raise AssertionError('Hidden variant must bypass the full vocabulary LM head')

    def save_pretrained(self,path):
        path.mkdir()
        (path/'adapter_model.safetensors').write_bytes(b'cpu-contract-placeholder')


class PRMCoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = NeuralPRM(TinyBackbone())
        self.batch = dict(input_ids=torch.tensor([[1,2,3,0],[2,3,4,5]]),
                          attention_mask=torch.tensor([[1,1,1,0],[1,1,1,1]]),
                          turn_end_indices=torch.tensor([[2,-1],[1,3]]),
                          turn_mask=torch.tensor([[True,False],[True,True]]),
                          labels=torch.tensor([.5,1.]))
        self.identity = dict(base_revision='固定版本',base_sha256='基础权重哈希',
                             tokenizer_sha256={'tokenizer.json':'哈希'},
                             initialization={'kind':'锁定基础模型'},
                             template_sha256='模板哈希',source_commit='源码提交')
    def test_exact_reference_loss_and_padding(self):
        with torch.no_grad():
            self.model.score_head.weight.fill_(.02)
            self.model.score_head.bias.fill_(.1)
        output = self.model(**self.batch)
        logits = self.model.backbone(self.batch['input_ids'],self.batch['attention_mask']).logits
        expected = torch.stack([self.model.score_head(logits[0,2]).sum(),
                                self.model.score_head(logits[1,[1,3]]).sum()])
        torch.testing.assert_close(output['totals'],expected)
        torch.testing.assert_close(output['loss'],(expected-self.batch['labels']).square().mean())
        self.assertEqual(float(output['step_scores'][0,1].detach()),0)
    def test_zero_head_first_step_then_backbone_gradient(self):
        opt = torch.optim.SGD(self.model.parameters(),lr=.01)
        optimizer_step(self.model,self.batch,opt)
        self.assertEqual(float(self.model.backbone.embedding.weight.grad.abs().sum()),0)
        self.assertGreater(float(self.model.score_head.weight.grad.abs().sum()),0)
        optimizer_step(self.model,self.batch,opt)
        self.assertGreater(float(self.model.backbone.embedding.weight.grad.abs().sum()),0)

    def test_accumulation_matches_one_logical_batch_before_clipping(self):
        with torch.no_grad():
            self.model.score_head.weight.fill_(.04)
        together = copy.deepcopy(self.model)
        batch = {key: tensor.repeat((4,) + (1,) * (tensor.ndim - 1))
                 for key, tensor in self.batch.items()}
        # Uneven sizes distinguish weighting by trajectory from weighting by batch.
        microbatches = [{key: value[part] for key, value in batch.items()}
                        for part in (slice(0, 1), slice(1, 3), slice(3, 8))]
        expected = optimizer_step(together, batch, torch.optim.SGD(together.parameters(), lr=.1), .15)
        opt = torch.optim.SGD(self.model.parameters(), lr=.1)
        with patch.object(opt, 'step', wraps=opt.step) as step, patch.object(
                torch.nn.utils, 'clip_grad_norm_', wraps=torch.nn.utils.clip_grad_norm_) as clip:
            actual = accumulated_optimizer_step(self.model, microbatches, opt, .15)
        step.assert_called_once()
        clip.assert_called_once()
        self.assertGreater(expected['gradient_norm'], .15)
        self.assertAlmostEqual(actual['gradient_norm'], expected['gradient_norm'], places=5)
        self.assertAlmostEqual(actual['loss'], expected['loss'], places=6)
        torch.testing.assert_close(torch.tensor(actual['totals_before_update']),
                                   torch.tensor(expected['totals_before_update']))
        for actual_param, expected_param in zip(self.model.parameters(), together.parameters()):
            torch.testing.assert_close(actual_param, expected_param)

    def test_invalid_later_microbatch_aborts_before_updating(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=.01)
        before = [p.detach().clone() for p in self.model.parameters()]
        invalid = dict(self.batch, labels=torch.tensor([float('nan'), 0.]))
        with self.assertRaises(ValueError):
            accumulated_optimizer_step(self.model, [self.batch, invalid], opt)
        self.assertFalse(opt.state)
        for p, original in zip(self.model.parameters(), before):
            torch.testing.assert_close(p, original, atol=0, rtol=0)
    def test_bad_boundaries_are_rejected(self):
        for positions in [[[3,-1],[1,3]],[[2,-1],[3,1]],[[2,-1],[1,1]],[[8,-1],[1,3]]]:
            with self.subTest(positions=positions),self.assertRaises(ValueError):
                self.model(**dict(self.batch,turn_end_indices=torch.tensor(positions)))
    def test_labels_are_not_model_inputs(self):
        a=self.model(**self.batch)
        b=self.model(**dict(self.batch,labels=torch.tensor([1.,0.])))
        torch.testing.assert_close(a['step_scores'],b['step_scores'])
    def test_future_suffix_does_not_change_earlier_score(self):
        with torch.no_grad():self.model.score_head.weight.fill_(.1)
        a=self.model(**self.batch)
        ids=self.batch['input_ids'].clone();ids[1,2:]=torch.tensor([6,6])
        b=self.model(**dict(self.batch,input_ids=ids))
        torch.testing.assert_close(a['step_scores'][1,0],b['step_scores'][1,0])
    def test_save_head_reload_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint'
            save_prm(self.model,path,self.identity)
            with self.assertRaises(ValueError):save_prm(self.model,path,self.identity)
            with torch.no_grad():self.model.score_head.weight.fill_(3)
            restore_head(self.model,path,self.identity)
            self.assertEqual(float(self.model.score_head.weight.detach().abs().sum()),0)
            with self.assertRaises(ValueError):verify_prm(path,dict(self.identity,base_revision='错误版本'))
            (path/'head.pt').write_bytes('损坏'.encode('utf-8'))
            with self.assertRaises(ValueError):verify_prm(path,self.identity)

    def test_hidden_representation_scoring_and_versioned_reload(self):
        with self.assertRaises(ValueError):
            NeuralPRM(TinyHiddenPeft(),'final_hidden_state')
        hidden=NeuralPRM(TinyHiddenPeft(),'final_hidden_state',checkpoint_schema_version=2)
        self.assertEqual(hidden.score_head.weight.shape,(1,5))
        with torch.no_grad():
            hidden.score_head.weight.fill_(.1)
            hidden.score_head.bias.fill_(.2)
        out=hidden(**self.batch)
        h=hidden.backbone.get_base_model().model(
            self.batch['input_ids'],self.batch['attention_mask']).last_hidden_state
        expected=torch.stack([
            hidden.score_head(h[0,2]).sum(),
            hidden.score_head(h[1,[1,3]]).sum()])
        torch.testing.assert_close(out['totals'],expected)
        torch.testing.assert_close(out['loss'],(expected-self.batch['labels']).square().mean())
        self.assertEqual(float(out['step_scores'][0,1].detach()),0.)
        other=hidden(**dict(self.batch,labels=torch.tensor([0.,0.])))
        torch.testing.assert_close(out['step_scores'],other['step_scores'])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'hidden-checkpoint'
            save_prm(hidden,path,self.identity)
            manifest=json.loads((path/'prm-manifest.json').read_text())
            self.assertEqual(manifest['schema_version'],2)
            self.assertEqual(manifest['architecture']['representation'],'final_hidden_state')
            self.assertEqual(manifest['architecture']['input_dim'],5)
            verify_prm(path,self.identity,'final_hidden_state',7,5)
            with self.assertRaises(ValueError):
                verify_prm(path,self.identity,'vocab_logits',7,5)
            with self.assertRaises(ValueError):
                verify_prm(path,self.identity,'final_hidden_state',7,6)
            with torch.no_grad():
                hidden.score_head.weight.fill_(4.)
            restore_head(hidden,path,self.identity)
            torch.testing.assert_close(hidden(**self.batch)['totals'],expected)

    def test_paired_v2_logits_has_distinct_manifest_from_hidden(self):
        logits=NeuralPRM(TinyHiddenPeft(),'vocab_logits',checkpoint_schema_version=2)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'logits-checkpoint'
            save_prm(logits,path,self.identity)
            manifest=json.loads((path/'prm-manifest.json').read_text())
            self.assertEqual(manifest['schema_version'],2)
            self.assertEqual(manifest['architecture']['representation'],'vocab_logits')
            verify_prm(path,self.identity,'vocab_logits',7,5)
            with self.assertRaises(ValueError):
                verify_prm(path,self.identity,'final_hidden_state',7,5)


if __name__ == '__main__':unittest.main()
