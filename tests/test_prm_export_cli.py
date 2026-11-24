"""真实data CLI控制流：技术中断/超长不能重新进入training-jsonl。"""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import types
import sys
import unittest
from unittest.mock import patch
from webshop_lab.prm import data
from test_prm_data import episode


class ExportCLITests(unittest.TestCase):
    def run_case(self,reason):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'raw.jsonl';split=root/'split';split.mkdir()
            rows=[episode(i,'request '+str(i)) for i in range(1,5)]
            if reason=='technical_interruption':rows[0]['termination']='environment_error'
            src.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            for name,ids in [('train',[1,2,3,4]),('val',[5]),('test',[6])]:
                (split/(name+'_ids.json')).write_text(json.dumps(ids))
            goal=root/'goals.json';goal.write_text(json.dumps([
                {'task_id':i,'instruction':'request '+str(i)} for i in range(1,7)]))
            cfg=types.SimpleNamespace(load_in_4bit=False,tokenizer='local',tokenizer_revision='r',
                max_length=2048,sft_adapter_files_sha256={'adapter_model.safetensors':'a'*64},
                fingerprint=lambda:'f'*64)
            tok=types.SimpleNamespace(chat_template='template')
            def encode(tokenizer,messages,max_length):
                if reason=='too_long' and 'request 1' in messages[0]['content']:
                    raise data.TrajectoryTooLong('too long')
                return {'turn_end_indices':[3],'input_ids':[1,2,3,4],'template_sha256':'t'}
            argv=['--rollout',str(src),'--config','ignored','--split-dir',str(split),
                  '--goal-map',str(goal),'--output',str(root/'ds'),'--fit-fraction','0.5',
                  '--training-jsonl','fit='+str(root/'fit.training.jsonl'),
                  '--training-jsonl','dev='+str(root/'dev.training.jsonl')]
            with patch('webshop_lab.prm.config.load_config',return_value=cfg), \
                 patch.dict(sys.modules,{'transformers':types.SimpleNamespace(
                     AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**kw:tok))}), \
                 patch.object(data,'tokenize_trajectory',side_effect=encode), \
                 patch('sys.stdout',new_callable=io.StringIO):
                data.main(argv)
            manifest=json.loads((root/'ds/manifest.json').read_text())
            exported=[]
            for name in ('fit','dev'):
                target=root/(name+'.training.jsonl');payload=target.read_bytes()
                samples=[json.loads(s) for s in payload.splitlines()]
                encoded=[json.loads(s) for s in (root/'ds'/(name+'.jsonl')).read_text().splitlines()]
                expected=[[r['task_id'],r['repeat']] for r in encoded]
                self.assertEqual([[r['task_id'],r['repeat']] for r in samples],expected)
                self.assertEqual(manifest['training_files'][name]['sample_keys'],expected)
                self.assertEqual(manifest['training_files'][name]['sha256'],hashlib.sha256(payload).hexdigest())
                self.assertEqual(len(samples),manifest['files'][name]['samples'])
                exported.extend(r['task_id'] for r in samples)
            self.assertNotIn(1,exported)
            self.assertEqual(sorted(exported),[2,3,4])
            self.assertEqual(manifest['dropped'][0]['reason'],reason)

    def test_valid_and_overlong_keep_same_manifest_and_training_set(self):self.run_case('too_long')
    def test_valid_and_technical_interruption_complete_without_export_error(self):self.run_case('technical_interruption')
