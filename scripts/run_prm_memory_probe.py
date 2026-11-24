"""Bounded eight-trajectory memorization probe using the unchanged PRM trainer.

Only fit examples are used. Passing this test does not establish generalization.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


def memory_metrics(predictions, labels):
    ids=[r['task_id'] for r in predictions]
    if len(ids)!=len(set(ids)) or set(ids)!=set(labels):
        raise ValueError('Missing, duplicated or unexpected memory-test prediction')
    rows=[]
    for r in predictions:
        p=float(r['total']);y=float(labels[r['task_id']])
        if not math.isfinite(p) or not math.isfinite(y):
            raise ValueError('Nonfinite memory-test score or label')
        rows.append({'task_id':r['task_id'],'prediction':p,'label':y,'absolute_error':abs(p-y)})
    mse=statistics.mean((r['prediction']-r['label'])**2 for r in rows)
    maximum=max(r['absolute_error'] for r in rows)
    return {'mse':mse,'max_absolute_error':maximum,'passed':mse<0.001 and maximum<0.1,'samples':rows}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(plan, output):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prm_reload_validation import compare_predictions
    from webshop_lab.prm.config import load_config
    from webshop_lab.prm.runtime import build_prm
    from webshop_lab.prm.train import load_samples,predict
    cfg=load_config(output/'config.json')
    model,tok,_,_=build_prm(cfg,checkpoint=output,device='cuda',training=False)
    ids=set(json.loads(Path(plan['train_ids']).read_text()))
    rows,rejected,_=load_samples(Path(plan['samples']),tok,cfg.max_length,ids,True,
                                cfg.sft_adapter_files_sha256['adapter_model.safetensors'])
    if rejected or len(rows)!=8:
        raise ValueError('Memory probe must retain exactly eight complete inputs')
    predictions=predict(model,tok,rows)
    reference=json.loads((output/'reference-predictions.json').read_text())
    labels={int(k):v for k,v in plan['labels'].items()}
    comparison=compare_predictions(predictions,reference,list(labels),atol=1e-5,rtol=0.0)
    metrics=memory_metrics(predictions,labels)
    with (output/'memory-independent-reload.json').open('x') as stream:
        json.dump({'metrics':metrics,'max_step_difference':comparison['max_step_difference'],
                   'reload_comparison':comparison,'predictions':predictions},stream,indent=2,allow_nan=False)
    print(json.dumps({'metrics':metrics,'reload_comparison':comparison},allow_nan=False),flush=True)


def run(plan_path):
    import fcntl
    plan=json.loads(plan_path.read_text())
    root=plan_path.parent
    if (root/'progress.json').exists():
        raise ValueError('Existing probe attempt; no overwrite/restart')
    locks=[]
    for name in ['gpu-training.lock','fp32-validation.lock']:
        f=open(Path(plan['lock_dir'])/name,'a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    env={**os.environ,'PYTHONPATH':str(Path(plan['source'])/'src'),
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    state={'status':'running','completed_updates':0,'max_updates':plan['max_updates'],
           'scope':'fit-only memorization diagnostic; no dev or test','history':[]}
    labels={int(k):v for k,v in plan['labels'].items()}
    previous=None;parent=None

    def save():
        p=root/'progress.tmp.json';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(root/'progress.json')

    def verify():
        helper=str(Path(__file__).resolve().with_name('prm_reload_validation.py'))
        if plan['input_sha256'].get(helper)!=sha(helper):
            raise ValueError('Freeze the reload validator SHA in a new attempt plan; preserve old plans')
        for p,h in plan['input_sha256'].items():
            if sha(p)!=h:raise ValueError('Frozen probe input changed: '+p)
        revision=subprocess.check_output(['git','-C',plan['source'],'rev-parse','HEAD'],text=True).strip()
        status=subprocess.check_output(['git','-C',plan['source'],'status','--porcelain','--untracked-files=no'],text=True)
        if revision!=plan['source_commit'] or status.strip():raise ValueError('Execution source changed')

    try:
        verify();save()
        for target in range(plan['interval'],plan['max_updates']+1,plan['interval']):
            output=root/('step-'+str(target))
            cfg=json.loads(Path(plan['initial_config']).read_text());cfg['output_dir']=str(output)
            config=root/('step-'+str(target)+'.config.json')
            with config.open('x') as stream:json.dump(cfg,stream,indent=2)
            command=[plan['python'],'-u','-m','webshop_lab.prm.train','--config',str(config),
                     '--samples',plan['samples'],'--train-ids',plan['train_ids'],'--steps',str(plan['interval']),
                     '--learning-rate',str(plan['learning_rate']),'--weight-decay','0.01','--seed','42','--device','cuda']
            if previous is not None:command+=['--resume',str(previous)]
            state['current_target_updates']=target;save();verify()
            with (root/('step-'+str(target)+'.command.json')).open('x') as stream:
                json.dump({'argv':command,'cwd':plan['cwd']},stream,indent=2)
            print('TRAIN_TO',target,flush=True)
            with (root/('step-'+str(target)+'.log')).open('xb') as log:
                subprocess.run(command,cwd=plan['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            summary=json.loads((output/'summary.json').read_text());verify()
            if (summary['optimizer_steps']!=target or summary['retained_samples']!=8 or summary['rejected'] or
                    not summary['head_changed'] or not summary['lora_changed']):
                raise ValueError('Missing actual updates, full inputs, or parameter changes')
            if parent is None:
                if summary['parameters_before']!=plan['initial_parameter_digests']:
                    raise ValueError('Not initialized from common B1/zero head')
            elif summary['parameters_before']!=parent['parameters_after']:
                raise ValueError('Saved parameter chain discontinuity')
            predictions=json.loads((output/'reference-predictions.json').read_text())
            metrics=memory_metrics(predictions,labels)
            state['completed_updates']=target;state['latest_checkpoint']=str(output)
            state['history'].append({'updates':target,**metrics})
            print('FIT_METRICS',json.dumps(state['history'][-1]),flush=True);save()
            previous,parent=output,summary
            if metrics['passed']:break
        with (root/'independent-reload.log').open('xb') as log:
            subprocess.run([plan['python'],'-u',str(Path(__file__).resolve()),'--plan',str(plan_path),
                            '--evaluate',str(previous)],cwd=plan['cwd'],env=env,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        reloaded=json.loads((previous/'memory-independent-reload.json').read_text())
        state['independent_reload_metrics']=reloaded['metrics']
        state['status']='complete';state['memory_test_passed']=reloaded['metrics']['passed']
        if state['memory_test_passed']:
            state['stop_reason']='fit threshold met and independently reloaded'
        elif target==plan['max_updates']:
            state['stop_reason']='fixed update cap reached without independently verified fit'
        else:
            state['stop_reason']='saved fit threshold met but independent reload fit threshold not met'
    except BaseException as error:
        state.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--evaluate',type=Path)
    args=parser.parse_args()
    if args.evaluate:evaluate(json.loads(args.plan.read_text()),args.evaluate)
    else:run(args.plan)
