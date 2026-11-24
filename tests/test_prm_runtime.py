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
from webshop_lab.prm.runtime import build_prm, tensor_batch, sha
from webshop_lab.prm.tokenize import tokenize_trajectory
from webshop_lab.prm.infer import compare_predictions
from webshop_lab.prm.train import load_samples

REV='989aa7980e4cf806f80c7fef2b1adb7bc71aa306'


class RuntimeTests(unittest.TestCase):
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
            messages=[{'role':'system','content':'Shop carefully.'},
                      {'role':'user','content':'Find black shoes.'},
                      {'role':'assistant','content':'Action: search[black shoes]'}]
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
            self.assertEqual(before,{n:sha(Path(cfg.sft_adapter)/n) for n in before})
            # 验证续训保留优化器步数，且使用新输出目录。
            newer=cfg.to_dict();newer['output_dir']=str(root/'continued')
            config2=root/'config2.json';config2.write_text(json.dumps(newer))
            resumed=train.copy();resumed[resumed.index(str(config))]=str(config2)
            resumed[resumed.index('3')]='1';resumed += ['--resume',cfg.output_dir]
            third=subprocess.run(resumed,capture_output=True,text=True,env=env,timeout=120)
            self.assertEqual(third.returncode,0,third.stderr)
            self.assertEqual(json.loads((root/'continued/summary.json').read_text())['optimizer_steps'],4)

    def test_hidden_v2_real_qwen_train_and_new_process_reload(self):
        # Only runs when real pinned Qwen tokenizer + CPU Torch/Transformers/PEFT exist.
        # This checks small random Qwen integration, NOT the 1.5B model or GPU.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            original=self.make_fixture(root)
            config_data=original.to_dict()
            config_data.update(schema_version='prm-qwen-config-v2',
                               representation='final_hidden_state')
            config=root/'hidden-config.json'
            config.write_text(json.dumps(config_data))
            messages=[{'role':'system','content':'Shop carefully.'},
                      {'role':'user','content':'Find blue shoes.'},
                      {'role':'assistant','content':'Action: search[blue shoes]'}]
            sample=root/'hidden-samples.jsonl'
            sample.write_text(json.dumps({'task_id':111,'messages':messages,
                'terminal_reward':1.,'done':True,'termination':'purchase',
                'policy_adapter_sha256':original.sft_adapter_files_sha256['adapter_model.safetensors'],
                'source_sha256':'a'*64})+'\n')
            ids=root/'hidden-train-ids.json';ids.write_text('[111]')
            env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
                 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
            train=[sys.executable,'-m','webshop_lab.prm.train','--config',str(config),
                   '--samples',str(sample),'--train-ids',str(ids),'--steps','2','--device','cpu']
            trained=subprocess.run(train,capture_output=True,text=True,env=env,timeout=120)
            self.assertEqual(trained.returncode,0,trained.stderr)
            checkpoint=Path(config_data['output_dir'])
            manifest=json.loads((checkpoint/'prm-manifest.json').read_text())
            self.assertEqual(manifest['schema_version'],2)
            self.assertEqual(manifest['architecture']['representation'],'final_hidden_state')
            self.assertEqual(manifest['architecture']['input_dim'],8)
            infer=[sys.executable,'-m','webshop_lab.prm.infer','--config',str(config),
                   '--checkpoint',str(checkpoint),'--samples',str(sample),
                   '--output',str(root/'hidden-reloaded.json'),
                   '--reference',str(checkpoint/'reference-predictions.json'),'--device','cpu']
            reloaded=subprocess.run(infer,capture_output=True,text=True,env=env,timeout=120)
            self.assertEqual(reloaded.returncode,0,reloaded.stderr)
            self.assertTrue(json.loads((root/'hidden-reloaded.json').read_text())['reference_match'])
            wrong=dict(config_data,representation='vocab_logits',output_dir=str(root/'wrong-arm'))
            wrong_config=root/'wrong-config.json';wrong_config.write_text(json.dumps(wrong))
            wrong_cmd=infer.copy();wrong_cmd[wrong_cmd.index(str(config))]=str(wrong_config)
            wrong_cmd[wrong_cmd.index(str(root/'hidden-reloaded.json'))]=str(root/'wrong-output.json')
            rejected=subprocess.run(wrong_cmd,capture_output=True,text=True,env=env,timeout=120)
            self.assertNotEqual(rejected.returncode,0)
            self.assertFalse((root/'wrong-output.json').exists())

    def test_bridge_maps_only_valid_keys_and_masks(self):
        messages=[{'role':'system','content':'Shop.'},{'role':'user','content':'Page'},
                  {'role':'assistant','content':'Action: search[shoe]'}]
        row=tokenize_trajectory(self.tokenizer,messages,256)
        row['terminal_reward']=.5
        batch=tensor_batch([row],self.tokenizer.pad_token_id,'cpu',True)
        self.assertEqual(set(batch),{'input_ids','attention_mask','turn_end_indices','turn_mask','labels'})
        self.assertEqual(batch['turn_mask'].dtype,torch.bool)
        self.assertEqual(batch['turn_end_indices'].dtype,torch.long)
        self.assertNotIn('labels',tensor_batch([row],self.tokenizer.pad_token_id,'cpu',False))
    def test_reload_comparison_rejects_false_match(self):
        a=[{'task_id':1,'turn_end_indices':[2],'step_scores':[.3],'total':.3}]
        compare_predictions(a,a,1e-5,1e-5)
        b=[dict(a[0],step_scores=[.4])]
        with self.assertRaises(ValueError):compare_predictions(a,b,1e-5,1e-5)

    def test_train_pool_policy_identity_and_terminal_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'samples.jsonl'
            row={'task_id':1001,'messages':[{'role':'system','content':'Shop.'},
                {'role':'assistant','content':'Action: search[shoes]'}],
                'terminal_reward':0.,'done':False,'termination':'max_steps',
                'source_sha256':'a'*64,'policy_adapter_sha256':'b'*64}
            def check(value,allowed={1001},policy='b'*64):
                p.write_text(json.dumps(value)+'\n')
                return load_samples(p,self.tokenizer,256,allowed,True,policy)
            self.assertEqual(len(check(row)[0]),1)
            with self.assertRaises(ValueError):check(row,{1002})
            with self.assertRaises(ValueError):check(row,policy='c'*64)
            with self.assertRaises(ValueError):check(dict(row,termination='timeout'))
            with self.assertRaises(ValueError):check(dict(row,terminal_reward=.75))

    def test_b1_merge_is_preserved_before_new_prm_lora(self):
        from safetensors.torch import load_file
        with tempfile.TemporaryDirectory() as tmp:
            cfg=self.make_fixture(Path(tmp))
            original=load_file(str(Path(cfg.model)/'model.safetensors'))['model.layers.0.self_attn.q_proj.weight']
            model,_,_,_=build_prm(cfg,device='cpu',training=True)
            projection=model.backbone.get_base_model().model.layers[0].self_attn.q_proj
            self.assertFalse(torch.equal(original,projection.get_base_layer().weight.detach()))
            self.assertEqual(set(model.backbone.peft_config),{'default'})
            names=[n for n,p in model.named_parameters() if p.requires_grad]
            self.assertTrue(all('lora_' in n or n.startswith('score_head.') for n in names))


if __name__ == '__main__':unittest.main()
