"""小型真实Qwen/PEFT的CPU接线测试；不代表1.5B GPU验收。"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
try:
    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    from peft import LoraConfig, get_peft_model
except ImportError:
    raise unittest.SkipTest('本组测试需要目标兼容版本的Torch、Transformers和PEFT')
from webshop_lab.prm.config import PRMConfig
try:
    from webshop_lab.prm.runtime import build_prm, tensor_batch, sha
except ImportError:
    raise unittest.SkipTest('跨PR测试需要#43的runtime/core/train，不能把缺依赖当通过')
from webshop_lab.prm.tokenize import tokenize_trajectory
from webshop_lab.prm.infer import compare_predictions
from webshop_lab.prm.train import load_samples
from webshop_lab.prm.data import episode_messages

REV='989aa7980e4cf806f80c7fef2b1adb7bc71aa306'


class CrossRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',
                revision=REV,local_files_only=True,trust_remote_code=False)
        except OSError as error:
            raise unittest.SkipTest('需要本地缓存固定Qwen tokenizer，不下载底模') from error
        torch.set_num_threads(1)

    def make_fixture(self,root):
        base=root/'base';tok=root/'tokenizer';b1=root/'b1'
        self.tokenizer.save_pretrained(tok)
        torch.manual_seed(42)
        tiny=Qwen2ForCausalLM(Qwen2Config(vocab_size=151936,hidden_size=8,
            intermediate_size=16,num_hidden_layers=1,num_attention_heads=2,
            num_key_value_heads=1,max_position_embeddings=512,
            eos_token_id=self.tokenizer.eos_token_id,pad_token_id=self.tokenizer.pad_token_id))
        tiny.save_pretrained(base)
        sft=get_peft_model(tiny,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],
                                         task_type='CAUSAL_LM',bias='none'))
        with torch.no_grad():
            for name,p in sft.named_parameters():
                if 'lora_B' in name:p.fill_(.01)
        sft.save_pretrained(b1)
        cfg=PRMConfig(model=str(base),revision=REV,base_weights_sha256=sha(base/'model.safetensors'),
            tokenizer=str(tok),tokenizer_revision=REV,
            tokenizer_files_sha256={p.name:sha(p) for p in tok.iterdir() if p.is_file()},
            initialization='sft_b1',sft_adapter=str(b1),
            sft_adapter_files_sha256={n:sha(b1/n) for n in ['adapter_model.safetensors','adapter_config.json']},
            output_dir=str(root/'output'),max_length=256,lora_r=2,lora_alpha=4)
        return cfg

    def test_real_qwen_batch_adapter_training_and_fresh_process_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=self.make_fixture(root)
            config=root/'config.json';config.write_text(json.dumps(cfg.to_dict()))
            rollout={'task_id':1001,'instruction':'Find black shoes.',
                     'steps':[{'observation':'Page: shoes','response':'Action: search[black shoes]'}]}
            messages=episode_messages(rollout,require_labels=False)
            raw=root/'raw.jsonl';raw.write_text(json.dumps(rollout)+'\n')
            samples=root/'samples.jsonl'
            samples.write_text(json.dumps({'task_id':1001,'messages':messages,
                'terminal_reward':1.,'done':True,'termination':'purchase',
                'policy_adapter_sha256':cfg.sft_adapter_files_sha256['adapter_model.safetensors'],
                'source_sha256':'a'*64})+'\n')
            ids=root/'train_ids.json';ids.write_text('[1001]')
            before={n:sha(Path(cfg.sft_adapter)/n) for n in cfg.sft_adapter_files_sha256}
            env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
                 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
            train=[sys.executable,'-m','webshop_lab.prm.train','--config',str(config),
                   '--samples',str(samples),'--train-ids',str(ids),'--steps','3','--device','cpu']
            first=subprocess.run(train,capture_output=True,text=True,env=env,timeout=120)
            self.assertEqual(first.returncode,0,first.stderr)
            summary=json.loads((Path(cfg.output_dir)/'summary.json').read_text())
            self.assertTrue(summary['head_changed'])
            self.assertTrue(summary['lora_changed'])
            self.assertEqual(summary['records'][0]['lora_gradient_l1'],0)
            self.assertTrue(any(r['lora_gradient_l1']>0 for r in summary['records'][1:]))
            infer=[sys.executable,'-m','webshop_lab.prm.infer','--config',str(config),
                   '--checkpoint',cfg.output_dir,'--samples',str(samples),
                   '--output',str(root/'reloaded.json'),'--reference',
                   str(Path(cfg.output_dir)/'reference-predictions.json'),'--device','cpu']
            second=subprocess.run(infer,capture_output=True,text=True,env=env,timeout=120)
            self.assertEqual(second.returncode,0,second.stderr)
            self.assertTrue(json.loads((root/'reloaded.json').read_text())['reference_match'])
            # #43训练保存 -> #45原始rollout输入，独立子进程共用同一runtime与identity。
            raw_infer=infer.copy();raw_infer[raw_infer.index('--samples')]='--rollout'
            raw_infer[raw_infer.index(str(samples))]=str(raw)
            raw_infer[raw_infer.index(str(root/'reloaded.json'))]=str(root/'raw-reloaded.json')
            raw_infer+=['--task-id','1001']
            raw_result=subprocess.run(raw_infer,capture_output=True,text=True,env=env,timeout=120)
            self.assertEqual(raw_result.returncode,0,raw_result.stderr)
            self.assertEqual(json.loads((root/'raw-reloaded.json').read_text())['identity'],summary['identity'])
            self.assertTrue(json.loads((root/'raw-reloaded.json').read_text())['reference_match'])
            # 原始轨迹没有R/done等字段，验证推理无需标签。
            from webshop_lab.prm.model_core import verify_prm
            import dataclasses
            for kind in ('b1','tokenizer','head'):
                if kind=='b1':
                    changed=dataclasses.replace(cfg,sft_adapter_files_sha256={
                        **cfg.sft_adapter_files_sha256,'adapter_model.safetensors':'f'*64})
                    with self.assertRaises(ValueError):build_prm(changed,cfg.output_dir,'cpu',False)
                elif kind=='tokenizer':
                    changed=dataclasses.replace(cfg,tokenizer_files_sha256={
                        **cfg.tokenizer_files_sha256,'tokenizer.json':'f'*64})
                    with self.assertRaises(ValueError):build_prm(changed,cfg.output_dir,'cpu',False)
                else:
                    head=Path(cfg.output_dir)/'head.pt';original=head.read_bytes()
                    head.write_bytes(original+b'broken')
                    try:
                        with self.assertRaises(ValueError):verify_prm(cfg.output_dir,summary['identity'])
                    finally:head.write_bytes(original)

            self.assertEqual(before,{n:sha(Path(cfg.sft_adapter)/n) for n in before})
            # 验证续训保留优化器步数，且使用新输出目录。
            newer=cfg.to_dict();newer['output_dir']=str(root/'continued')
            config2=root/'config2.json';config2.write_text(json.dumps(newer))
            resumed=train.copy();resumed[resumed.index(str(config))]=str(config2)
            resumed[resumed.index('3')]='1';resumed += ['--resume',cfg.output_dir]
            third=subprocess.run(resumed,capture_output=True,text=True,env=env,timeout=120)
            self.assertEqual(third.returncode,0,third.stderr)
            self.assertEqual(json.loads((root/'continued/summary.json').read_text())['optimizer_steps'],4)


if __name__=='__main__':unittest.main()
