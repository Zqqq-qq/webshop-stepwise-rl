"""纯CPU契约测试，不作为Qwen或GPU实训证据。"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
try:
    import torch
except ImportError:
    raise unittest.SkipTest('本组PRM测试需安装PyTorch；不作为GPU证据')
from torch import nn
from webshop_lab.prm.model_core import NeuralPRM, save_prm, restore_head, verify_prm
from webshop_lab.prm.train_core import optimizer_step


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


if __name__ == '__main__':unittest.main()
