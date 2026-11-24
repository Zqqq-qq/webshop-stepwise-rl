"""Frozen three-arm diagnostic, not a full-coverage or held-out experiment.

Runs the unchanged project trainer, then independently reloads each checkpoint.
Only the input ordering and learning rate differ across arms. No early stopping.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def assess(plan, arm):
    # Sibling import also works when this script is copied to the device folder.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prm_reload_validation import compare_predictions, validate_predictions
    import torch
    from webshop_lab.prm.config import load_config
    from webshop_lab.prm.runtime import build_prm, tensor_batch
    from webshop_lab.prm.train import load_samples
    cfg=load_config(arm['config'])
    model,tok,_,_=build_prm(cfg,checkpoint=Path(arm['output']),device='cuda',training=False)
    allowed=set(json.loads(Path(plan['train_ids']).read_text()))
    policy=cfg.sft_adapter_files_sha256['adapter_model.safetensors']
    predictions={}
    expected_ids={}
    for split in ('fit','dev'):
        expected_ids[split]=[row['task_id'] for row in read_rows(Path(plan[split]))]
        rows,rejected,_=load_samples(Path(plan[split]),tok,cfg.max_length,allowed,True,policy)
        if rejected:
            raise ValueError('Diagnostic inputs unexpectedly rejected')
        result=[]
        for task_id,row in rows:
            batch=tensor_batch([row],tok.pad_token_id,'cuda',False)
            with torch.inference_mode():
                out=model(**batch)
            result.append({'task_id':task_id,'prediction':float(out['totals'][0]),
                           'total':float(out['totals'][0]),
                           'label':row['terminal_reward'],'turns':len(row['turn_end_indices']),
                           'turn_end_indices':row['turn_end_indices'],
                           'step_scores':out['step_scores'][0].cpu().tolist()})
        validate_predictions(result,expected_ids[split])
        predictions[split]=result
    fit=predictions['fit'];dev=predictions['dev']
    mean=statistics.mean(r['label'] for r in fit)
    length_means={t:statistics.mean(r['label'] for r in fit if r['turns']==t)
                  for t in {r['turns'] for r in fit}}
    metrics={'scope':'exploratory diagnostic; previously viewed train-pool dev; no test',
             'fit_mse':statistics.mean((r['prediction']-r['label'])**2 for r in fit),
             'dev_mse':statistics.mean((r['prediction']-r['label'])**2 for r in dev),
             'constant_mse':statistics.mean((mean-r['label'])**2 for r in dev),
             'length_mse':statistics.mean((length_means.get(r['turns'],mean)-r['label'])**2 for r in dev)}
    for split,rows in predictions.items():
        p=[r['prediction'] for r in rows]; t=[r['turns'] for r in rows]
        metrics[split+'_prediction_turns_correlation']=statistics.correlation(p,t) if len(set(p))>1 and len(set(t))>1 else None
    reference=json.loads((Path(arm['output'])/'reference-predictions.json').read_text())
    comparison=compare_predictions(fit,reference,expected_ids['fit'],atol=1e-5,rtol=0.0)
    metrics['independent_reload_max_step_difference']=comparison['max_step_difference']
    metrics['independent_reload_max_total_difference']=comparison['max_total_difference']
    metrics['independent_reload_comparison']=comparison
    target=Path(arm['output'])/'diagnostic-evaluation.json'
    with target.open('x') as stream:
        json.dump({'metrics':metrics,'predictions':predictions},stream,indent=2,allow_nan=False)
    print(json.dumps(metrics),flush=True)


def run(plan_path):
    plan=json.loads(Path(plan_path).read_text())
    root=Path(plan_path).parent
    state_path=root/'progress.json'
    if state_path.exists():
        raise ValueError('Existing diagnostic progress; do not restart over evidence')
    locks=[]
    for name in ('gpu-training.lock','fp32-validation.lock'):
        lock=open(Path(plan['lock_dir'])/name,'a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        locks.append(lock)
    env={**os.environ,'PYTHONPATH':str(Path(plan['source'])/'src'),
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    state={'status':'running','completed_arms':[],'test_authorization':False}

    def save():
        tmp=root/'progress.tmp.json';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(state_path)

    def verify_inputs():
        helper=str(Path(__file__).resolve().with_name('prm_reload_validation.py'))
        if plan['input_sha256'].get(helper)!=digest(helper):
            raise ValueError('Freeze the reload validator SHA in a new attempt plan; preserve old plans')
        for path,expected in plan['input_sha256'].items():
            if digest(path)!=expected:
                raise ValueError('Frozen diagnostic input changed: '+path)
        revision=subprocess.check_output(['git','-C',plan['source'],'rev-parse','HEAD'],text=True).strip()
        clean=subprocess.check_output(['git','-C',plan['source'],'status','--porcelain','--untracked-files=no'],text=True)
        if revision!=plan['source_commit'] or clean.strip():
            raise ValueError('Execution source revision/cleanliness changed')

    try:
        verify_inputs()
        for arm in plan['arms']:
            state['current']=arm['name'];save()
            verify_inputs()
            command=[plan['python'],'-u','-m','webshop_lab.prm.train',
                     '--config',arm['config'],'--samples',arm['samples'],
                     '--train-ids',plan['train_ids'],'--steps',str(plan['steps']),
                     '--device','cuda','--learning-rate',str(arm['learning_rate']),
                     '--weight-decay','0.01','--seed','42']
            for phase,cmd in [('train',command),('evaluate',[plan['python'],'-u',str(Path(__file__).resolve()),
                                         '--plan',str(plan_path),'--evaluate',arm['name']])]:
                with (root/(arm['name']+'-'+phase+'.log')).open('xb') as log:
                    subprocess.run(cmd,cwd=plan['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            verify_inputs()
            summary=json.loads((Path(arm['output'])/'summary.json').read_text())
            if summary['optimizer_steps']!=plan['steps'] or not summary['head_changed'] or not summary['lora_changed']:
                raise ValueError('Missing actual optimizer updates/parameter changes')
            state['completed_arms'].append(arm['name']);save()
        state['status']='complete'
    except BaseException as error:
        state.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--evaluate')
    args=parser.parse_args()
    if args.evaluate:
        p=json.loads(args.plan.read_text());assess(p,next(a for a in p['arms'] if a['name']==args.evaluate))
    else:
        run(args.plan)
