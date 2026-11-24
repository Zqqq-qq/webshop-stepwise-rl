"""小型真实Qwen2/PEFT CPU表示探针；不是完整hidden runtime或1.5B/GPU验收。"""
import tempfile
import unittest
from pathlib import Path
try:
    import torch
    from torch import nn
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from peft import LoraConfig,get_peft_model
except ImportError:
    raise unittest.SkipTest('表示探针需要Torch/Transformers/PEFT，不把缺依赖当通过')


class VariantStructureTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(42);torch.set_num_threads(1)
        base=Qwen2ForCausalLM(Qwen2Config(vocab_size=64,hidden_size=8,intermediate_size=16,
              num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,attention_dropout=0.))
        return get_peft_model(base,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],
                    lora_dropout=0.,task_type='CAUSAL_LM')).float()

    def test_final_norm_hidden_maps_to_raw_logits_at_same_positions(self):
        m=self.model().eval();ids=torch.tensor([[1,2,3,4]])
        h=m.get_base_model().model(input_ids=ids,use_cache=False).last_hidden_state
        z=m(input_ids=ids,use_cache=False).logits
        torch.testing.assert_close(m.get_output_embeddings()(h[:,[1,3]]),z[:,[1,3]])
        self.assertEqual(h.shape[-1],8);self.assertEqual(z.shape[-1],64)

    def test_factored_logit_linear_head_has_equal_feature_gradient(self):
        torch.manual_seed(42)
        h=torch.randn(2,3,8,requires_grad=True);W=torch.randn(64,8);w=torch.randn(64);b=torch.tensor(.2)
        a=(h@W.T)@w+b;c=h@(W.T@w)+b
        torch.testing.assert_close(a,c,atol=1e-5,rtol=1e-5)
        ga=torch.autograd.grad(a.sum(),h,retain_graph=True)[0];gc=torch.autograd.grad(c.sum(),h)[0]
        torch.testing.assert_close(ga,gc,atol=1e-5,rtol=1e-5)

    def test_zero_hidden_head_then_lora_gradient_and_finite_head_roundtrip(self):
        m=self.model();head=nn.Linear(8,1);nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        opt=torch.optim.SGD(list(m.parameters())+list(head.parameters()),lr=.01)
        norms=[];ids=torch.tensor([[1,2,3,4]])
        for _ in range(2):
            opt.zero_grad();h=m.get_base_model().model(input_ids=ids,use_cache=False).last_hidden_state
            prediction=head(h[:,[1,3]]).sum();loss=(prediction-.75).square();loss.backward()
            norms.append(sum(float(p.grad.abs().sum()) for n,p in m.named_parameters() if 'lora_' in n and p.grad is not None))
            opt.step()
        self.assertEqual(norms[0],0.);self.assertGreater(norms[1],0.)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'head.pt';torch.save(head.state_dict(),p)
            restored=nn.Linear(8,1);restored.load_state_dict(torch.load(p,weights_only=True),strict=True)
            torch.testing.assert_close(restored(h.detach()),head(h.detach()))
            with self.assertRaises(RuntimeError):nn.Linear(64,1).load_state_dict(torch.load(p,weights_only=True),strict=True)
