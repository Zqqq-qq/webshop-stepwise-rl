"""Independent saved-artifact arithmetic and real-tokenizer review; no model run."""
from pathlib import Path, PurePosixPath
import hashlib
import io
import json
import math
import pickletools
import struct
import argparse
import tarfile
import zipfile
import numpy as np

if not __debug__:
    raise RuntimeError('Run this evidence audit without Python -O; assertions are integrity gates')

from transformers import AutoTokenizer

p=argparse.ArgumentParser(description=__doc__)
for name in ('bundle','previous-bundle','tokenizer','train-ids','output'):
    p.add_argument('--'+name,required=True)
args=p.parse_args()
archive=Path(args.bundle)
if Path(args.output).exists(): p.error('Output must be new')
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
assert digest=='b711fc49d9934adc321236a8837932406b6d1f3c8b81f8952d1f2c4a382453c8'
z=zipfile.ZipFile(archive); prefix='prm-evidence-20261124/'
def read(name): return z.read(prefix+name)
def j(name): return json.loads(read(name))
checks=[]
for line in read('SHA256SUMS').decode().splitlines():
    expected,name=line.split(None,1); name=name.strip().lstrip('*')
    assert not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts
    assert hashlib.sha256(read(name)).hexdigest()==expected,name
    checks.append(name)
plan=j('memory-8/plan.json'); progress=j('memory-8/progress.json')
rows=[json.loads(s) for s in read('memory-8/samples.jsonl').splitlines() if s]
ids=[r['task_id'] for r in rows]; labels={r['task_id']:r['terminal_reward'] for r in rows}
assert len(set(ids))==len(ids)==8
assert {int(k):v for k,v in plan['labels'].items()}==labels
assert list(plan['input_sha256'].values())[0]==hashlib.sha256(read('memory-8/samples.jsonl')).hexdigest()
tok=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True,trust_remote_code=False)
identity=j('memory-8/step-512/prm-manifest.json')['identity']
assert hashlib.sha256(tok.chat_template.encode()).hexdigest()==identity['template_sha256']
for name,d in identity['tokenizer_sha256'].items():
    assert hashlib.sha256((Path(args.tokenizer)/name).read_bytes()).hexdigest()==d
def encode(messages):
    ids=list(tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=False))
    assert len(ids)<=2048
    ends=[]
    for i,m in enumerate(messages):
        if m['role']!='assistant': continue
        prefix_ids=list(tok.apply_chat_template(messages[:i+1],tokenize=True,add_generation_prompt=False))
        assert ids[:len(prefix_ids)]==prefix_ids
        end=max(k for k,value in enumerate(prefix_ids) if value==151645)
        ends.append(end)
    return {'input_ids':ids,'turn_end_indices':ends}
encoded=[encode(r['messages']) for r in rows]
baseids=json.loads(Path(args.train_ids).read_text())
assert set(ids)<=set(baseids)
assert hashlib.sha256(Path(args.previous_bundle).read_bytes()).hexdigest()=='e1578d582bdc1deaa5954e0b33e869d137eba0cb8d3efd7c57ecd7fcdadf2ec7'
old=zipfile.ZipFile(args.previous_bundle)
fit=[json.loads(s) for s in old.read('diagnostic/fit-sorted.jsonl').splitlines() if s]
fitkeys={r['task_id']:r for r in fit}
assert all(r==fitkeys[r['task_id']] for r in rows)
source=tarfile.open(fileobj=io.BytesIO(old.read('execution-source.tar')))
source_files={m.name:source.extractfile(m).read() for m in source.getmembers()
              if m.isfile() and '/prm/' in m.name and m.name.endswith('.py')}
history=[]; missing=[]; source_mismatches=[]; step_records=[]; dtype_counts={}
for h in progress['history']:
    step=h['updates']; folder=f'memory-8/step-{step}'
    refs=j(folder+'/reference-predictions.json')
    assert [r['task_id'] for r in refs]==ids
    assert [r['task_id'] for r in h['samples']]==ids
    for raw,enc,ref,record in zip(rows,encoded,refs,h['samples']):
        assert ref['turn_end_indices']==enc['turn_end_indices']
        assert len(ref['step_scores'])==len(enc['turn_end_indices'])
        assert abs(sum(ref['step_scores'])-ref['total'])<1e-6
        assert record['label']==raw['terminal_reward']
        assert record['prediction']==ref['total']
    errors=np.array([r['total']-labels[r['task_id']] for r in refs])
    mse=float(np.mean(errors**2)); maxerr=float(np.abs(errors).max())
    assert abs(mse-h['mse'])<1e-12 and abs(maxerr-h['max_absolute_error'])<1e-12
    passed=mse<.001 and maxerr<.1
    assert h['passed']==passed
    history.append({'updates':step,'mse':mse,'max_absolute_error':maxerr,'passed':passed})
    summary=j(folder+'/summary.json')
    assert summary['optimizer_steps']==step
    step_records.extend(summary['records'])
    manifest=j(folder+'/prm-manifest.json')
    for name,expected in manifest['files_sha256'].items():
        filename=prefix+folder+'/'+name
        if filename not in z.namelist():
            missing.append(folder+'/'+name)
        else: assert hashlib.sha256(z.read(filename)).hexdigest()==expected
    for name,expected in manifest['identity']['source_sha256'].items():
        found=[payload for p,payload in source_files.items() if p.endswith('/'+name)]
        if len(found)!=1 or hashlib.sha256(found[0]).hexdigest()!=expected:
            source_mismatches.append(name)
    payload=read(folder+'/adapter/adapter_model.safetensors')
    size=struct.unpack('<Q',payload[:8])[0]
    header=json.loads(payload[8:8+size])
    dtypes={v['dtype'] for k,v in header.items() if k!='__metadata__'}
    dtype_counts[str(step)]=sorted(dtypes)
assert not source_mismatches
assert [h['updates'] for h in history]==list(range(64,513,64))
assert sorted(r['optimizer_step'] for r in step_records)==list(range(1,513))
reload=j('memory-8/step-512/memory-independent-reload.json')
assert reload['predictions']==j('memory-8/step-512/reference-predictions.json')
saved_reload_max=max(abs(a-b) for x,y in zip(reload['predictions'],j('memory-8/step-512/reference-predictions.json'))
                     for a,b in zip(x['step_scores'],y['step_scores']))
assert saved_reload_max==reload['max_step_difference']==0
repr_result=j('representation-diagnostic/representation-diagnostic.json')
assert [r['task_id'] for r in repr_result['samples']]==ids
assert [r['label'] for r in repr_result['samples']]==[labels[i] for i in ids]
assert [r['tokens'] for r in repr_result['samples']]==[len(e['input_ids']) for e in encoded]
assert [r['turns'] for r in repr_result['samples']]==[len(e['turn_end_indices']) for e in encoded]
reported_s=np.array(repr_result['singular_values'])
reported_errors=np.array([r['fitted']-r['label'] for r in repr_result['per_sample']])
assert abs(reported_s[0]/reported_s[-1]-repr_result['condition_number'])<1e-12
assert abs(np.sqrt(np.mean(reported_errors**2))-repr_result['residual_rms'])<1e-12
head=zipfile.ZipFile(io.BytesIO(read('memory-8/step-512/head.pt')))
pickle_name=next(n for n in head.namelist() if n.endswith('data.pkl'))
globals_found=[arg for op,arg,_ in pickletools.genops(head.read(pickle_name)) if op.name=='GLOBAL']
out={'scope':'CPU saved-record arithmetic and actual pinned-tokenizer audit; no GPU inference',
     'archive_sha256':digest,'zip_file_count':len(z.namelist()),'top_checksum_payloads_verified':len(checks),
     'samples_sha256':hashlib.sha256(read('memory-8/samples.jsonl')).hexdigest(),
     'sample_ids':ids,'same_as_original64_fit':True,'in_frozen_train':True,
     'labels':[labels[i] for i in ids],'token_counts':[len(e['input_ids']) for e in encoded],
     'turn_counts':[len(e['turn_end_indices']) for e in encoded],
     'all_checkpoint_scoring_positions_retokenized':True,'all_512_update_records_present':True,
     'core_source_matches_previous_archived_a67db6a':True,
     'adapter_saved_dtypes_by_step':dtype_counts,
     'head_pickle_storage_globals_inspected_without_loading':globals_found,
     'checkpoints':history,'memory_gate_passed':False,
     'saved_reload_step_max_difference':saved_reload_max,'independent_model_reload_performed_here':False,
     'missing_checkpoint_payloads':missing,
     'representation':{'status':'reported_result_only_missing_F',
         'samples_labels_tokens_turns_match':True,'singular_values_independently_recomputed':False,
         'reported_spectrum_ratio_arithmetic_checked':True,'reported_residual_rms_arithmetic_checked':True,
         'aggregation_in_script':'torch float32 sum, then cast to float64',
         'script_uses_base_plus_merged_B1_without_new_PRM_LoRA':True,
         'missing':['F or step logits','actual extraction config file','extraction core source hash binding'],
         'budget_claim_validated':False},
     'gate_status':{'BF16_risk':'separate mechanism; saved LoRA here is F32',
                    'memory_training':'failed at 512 updates',
                    'fixed_representation':'pending feature artifact',
                    'generalization':'not evaluated by this fit-only run',
                    'step_attribution':'not evaluated', 'test':'not authorized'}}
dest=Path(args.output)
dest.parent.mkdir(parents=True,exist_ok=True)
dest.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
