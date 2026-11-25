"""Recompute frozen historical dev evidence; no model execution or model selection."""
import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from audit_prm_hidden_results import load_json, sha, verified_archive, prediction_map

EVAL_SHA = '6e8ac493b72af56c9b689e5fdaac8b7d91cdc5586e8cfa48776b21f11a5f0518'
DATA_SHA = {'fit': '03a6370bc2048ea0119331aa06969ae6bde18e4e7ba5db3decde108153792500',
            'dev': 'df1f71a7843a04a654172c853656ece8f0793f1954557d735a8cd5b200c93d76'}
STEP120_SHA = '90b1b1f769f21cfe8b44627ebf628f0c3d2f32b506d19fe7d922cc63585aa49f'


def ranks(values):
    order = np.argsort(values, kind='stable')
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return result


def correlation(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def metrics(prediction, label):
    error = prediction-label
    return dict(n=len(label), mse=float(np.mean(error**2)), mae=float(np.mean(abs(error))),
                max_abs_error=float(np.max(abs(error))), pearson_r=correlation(prediction,label),
                spearman_rho=correlation(ranks(prediction),ranks(label)))


def fit_baselines(turns, labels):
    """Only callers' fit arrays enter fitting; no dev labels accepted."""
    mean = float(np.mean(labels))
    groups = {int(t):float(np.mean(labels[turns == t])) for t in np.unique(turns)}
    slope = float(np.sum((turns-turns.mean())*(labels-labels.mean())) /
                  np.sum((turns-turns.mean())**2))
    return mean, groups, float(mean-slope*turns.mean()), slope


def bootstrap_delta(model, baseline, labels, seed=42, repeats=10000, return_draw_mean=False):
    delta=(model-labels)**2-(baseline-labels)**2
    rng=np.random.default_rng(seed)
    sampled=delta[rng.integers(0,len(labels),size=(repeats,len(labels)))].mean(axis=1)
    result=[float(delta.mean()), *map(float,np.quantile(sampled,[.025,.975]))]
    return (result,float(sampled.mean())) if return_draw_mean else result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('evaluation-archive','data-archive','step120-archive','tokenizer','output'):
        parser.add_argument('--'+key, required=True)
    args=parser.parse_args()
    if Path(args.output).exists(): parser.error('Output must be new')
    files,checks=verified_archive(args.evaluation_archive,EVAL_SHA)
    checkpoint,_=verified_archive(args.step120_archive,STEP120_SHA)
    manifest=load_json(checkpoint['checkpoint/prm-manifest.json'])
    for name,digest in manifest['files_sha256'].items():
        if sha(checkpoint['checkpoint/'+name]) != digest: raise ValueError('Checkpoint SHA mismatch')
    version=files['EXECUTED_VERSION.md'].decode()
    for key,name in [('adapter','adapter/adapter_model.safetensors'),('head','head.pt'),
                     ('training_state','training-state.pt')]:
        if 'checkpoint_'+key+'_sha256='+manifest['files_sha256'][name] not in version:
            raise ValueError('Evaluation checkpoint declaration mismatch')
    data={}
    with zipfile.ZipFile(args.data_archive) as archive:
        source_manifest=load_json(archive.read('MANIFEST.sha256.json'))
        for name,digest in source_manifest.items():
            if sha(archive.read(name)) != digest: raise ValueError('Source payload SHA mismatch')
        for split,digest in DATA_SHA.items():
            blob=archive.read('partial911-source/'+split+'1451.training.jsonl')
            if sha(blob) != digest: raise ValueError('Wrong original fit/dev file')
            data[split]=[load_json(line) for line in blob.splitlines() if line]
        historical=load_json(archive.read('previous500-audit/prm-audit-20261124-predictions.json'))
    task_sets=[{r['task_id'] for r in data[s]} for s in ('fit','dev')]
    instruction_sets=[{r['instruction_sha256'] for r in data[s]} for s in ('fit','dev')]
    if task_sets[0]&task_sets[1] or instruction_sets[0]&instruction_sets[1]:
        raise ValueError('Fit/dev overlap')
    if not {351,95,5478,3542,359,5031,0,6225} <= task_sets[0]:
        raise ValueError('Memory examples outside fit')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True,trust_remote_code=False)
    for name,digest in manifest['identity']['tokenizer_sha256'].items():
        if sha((Path(args.tokenizer)/name).read_bytes()) != digest: raise ValueError('Tokenizer SHA mismatch')
    if sha(tokenizer.chat_template.encode()) != manifest['identity']['template_sha256']:
        raise ValueError('Template SHA mismatch')
    samples={}; predictions={}; coverage={}
    supplied=load_json(files['metrics.json'])
    length_audit=load_json(files['prm-dev-length-audit-20261124.json'])
    csv_rows=list(csv.DictReader(io.StringIO(files['per-task.csv'].decode())))
    csv_by_key={(r['split'],int(r['task_id'])):r for r in csv_rows}
    if len(csv_rows)!=len(csv_by_key): raise ValueError('Duplicate per-task CSV row')
    for split in ('fit','dev'):
        raw=data[split]; kept=[]; rejected=[]; positions={}
        if len({r['task_id'] for r in raw})!=len(raw): raise ValueError('Duplicate original task')
        for index,row in enumerate(raw):
            messages=row['messages']
            ids=list(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False))
            if len(ids)>2048:
                rejected.append(dict(row=index,task_id=row['task_id'],
                    reason=f'Complete trajectory has {len(ids)} tokens; limit=2048',
                    terminal_reward=row['terminal_reward']))
                continue
            ends=[]
            for i,message in enumerate(messages):
                if message['role']!='assistant': continue
                prefix=list(tokenizer.apply_chat_template(messages[:i+1],tokenize=True,add_generation_prompt=False))
                if ids[:len(prefix)]!=prefix: raise ValueError('Non-prefix-stable template')
                ends.append(max(k for k,v in enumerate(prefix) if v==151645))
            kept.append(row); positions[row['task_id']]=ends
        labels={r['task_id']:r['terminal_reward'] for r in kept}
        report=load_json(files[split+'-predictions.json'])
        if (report['identity']!=manifest['identity'] or report['total_samples']!=len(raw) or
                report['rejected']!=rejected or report['test_authorization'] is not False):
            raise ValueError('Prediction identity/rejection metadata differs')
        pred=prediction_map(report['predictions'],labels,positions)
        for row in kept:
            if float(csv_by_key[split,row['task_id']]['label'])!=row['terminal_reward']:
                raise ValueError('CSV label differs from original')
        terms={r['task_id']:r['termination'] for r in raw}
        kept_terms=dict(Counter(r['termination'] for r in kept))
        rejected_terms=dict(Counter(terms[r['task_id']] for r in rejected))
        audit=length_audit[split+'1451']
        kept_labels=dict(Counter(str(round(float(r['terminal_reward']),6)) for r in kept))
        rejected_labels=dict(Counter(str(round(float(r['terminal_reward']),6)) for r in rejected))
        if (audit['retained_termination']!=kept_terms or audit['rejected_termination']!=rejected_terms or
                set(audit['rejected_task_ids'])!={r['task_id'] for r in rejected} or
                audit['retained_label_distribution']!=kept_labels or
                audit['rejected_label_distribution']!=rejected_labels or
                (audit['total'],audit['retained'],audit['rejected'])!=(len(raw),len(kept),len(rejected))):
            raise ValueError('Length audit coverage differs')
        samples[split]=kept; predictions[split]=pred
        coverage[split]=dict(raw=len(raw),retained=len(kept),rejected=len(rejected),
            scoring_positions=sum(map(len,positions.values())),retained_termination=kept_terms,
            rejected_termination=rejected_terms,retained_task_ids=list(labels),
            retained_label_distribution=kept_labels,rejected_label_distribution=rejected_labels)
    if set(coverage['dev']['retained_task_ids'])!={r['task_id'] for r in historical}:
        raise ValueError('Historical dev task identity differs')
    if set(csv_by_key)!={(s,r['task_id']) for s in samples for r in samples[s]}:
        raise ValueError('CSV does not cover exact prediction sets')
    turns={s:np.array([sum(m['role']=='assistant' for m in r['messages']) for r in rows])
           for s,rows in samples.items()}
    labels={s:np.array([r['terminal_reward'] for r in rows]) for s,rows in samples.items()}
    mean,groups,intercept,slope=fit_baselines(turns['fit'],labels['fit'])
    params=supplied['baseline_parameters']
    if (not np.isclose(mean,params['constant'],rtol=0,atol=1e-12) or
            not np.isclose(intercept,params['turn_count_ols']['intercept'],rtol=0,atol=1e-12) or
            not np.isclose(slope,params['turn_count_ols']['slope'],rtol=0,atol=1e-12) or
            groups!={int(float(k)):v for k,v in params['turn_count_group_means'].items()}):
        raise ValueError('Published fit-only baseline parameters differ')
    calculated={}; vectors={}; subgroups={}
    for split in samples:
        vectors[split]={'model':np.array([predictions[split][r['task_id']]['total'] for r in samples[split]]),
            'constant':np.full(len(samples[split]),mean),
            'turn_count_mean':np.array([groups.get(int(t),mean) for t in turns[split]]),
            'turn_count_ols':intercept+slope*turns[split]}
        for method,values in vectors[split].items():
            for r,value in zip(samples[split],values):
                if not np.isclose(float(csv_by_key[split,r['task_id']][method]),value,rtol=0,atol=1e-12):
                    raise ValueError('Per-task CSV prediction/baseline differs')
            actual=metrics(values,labels[split]); calculated.setdefault(method,{})[split]=actual
            for key,value in actual.items():
                if value is not None and not np.isclose(value,supplied['metrics'][method][split][key],rtol=0,atol=1e-10):
                    raise ValueError('Published metric differs: '+method+'/'+split+'/'+key)
        subgroups[split]={}
        for term in ('purchase','max_steps'):
            idx=np.array([r['termination']==term for r in samples[split]])
            subgroups[split][term]=metrics(vectors[split]['model'][idx],labels[split][idx])
            zero=idx & (labels[split]==0)
            subgroups[split][term+'_zero_reward']=metrics(vectors[split]['model'][zero],labels[split][zero])
        if split=='fit':
            memory=np.array([r['task_id'] in {351,95,5478,3542,359,5031,0,6225} for r in samples[split]])
            subgroups[split]['trained_memory8']=metrics(vectors[split]['model'][memory],labels[split][memory])
            subgroups[split]['other493']=metrics(vectors[split]['model'][~memory],labels[split][~memory])
    bootstrap_results={method:bootstrap_delta(vectors['dev']['model'],vectors['dev'][method],labels['dev'],
                                             return_draw_mean=True)
               for method in ('constant','turn_count_mean','turn_count_ols')}
    intervals={method:result[0] for method,result in bootstrap_results.items()}
    # Each baseline reuses the same task-index draws from a fresh seed-42 generator.
    matched_intervals=intervals
    published=supplied['dev_paired_bootstrap_mse_delta_model_minus_baseline']
    if not all(np.allclose(matched_intervals[k][1:],published[k][1:],rtol=0,atol=1e-12) and
               np.isclose(bootstrap_results[k][1],published[k][0],rtol=0,atol=1e-12) for k in intervals):
        raise ValueError('Published paired bootstrap intervals not reproduced')
    output=dict(scope='Historical dev re-evaluation only; not formal generalization acceptance',
        evaluation_archive_sha256=EVAL_SHA,verified_payloads=len(checks),data_sha256=DATA_SHA,
        checkpoint_archive_sha256=STEP120_SHA,task_and_instruction_overlap=0,
        historical_dev_ids_match=True,coverage=coverage,baseline_fit_scope='retained fit 501 only',
        baseline_parameters=dict(constant=mean,turn_count_group_means=groups,
                                 turn_count_ols=dict(intercept=intercept,slope=slope)),
        metrics=calculated,model_termination_subgroups=subgroups,
        dev_bootstrap_model_minus_baseline=matched_intervals,
        published_delta_correction={k:dict(observed_mse_delta=intervals[k][0],
            published_first_value=published[k][0],bootstrap_replicate_mean=bootstrap_results[k][1])
            for k in intervals},
        bootstrap=dict(repeats=10000,seed=42,unit='task',rng='NumPy default_rng, reset per baseline'),
        caveat='Constant prediction correlation is mathematically undefined, reported as null here',
        model_executed_here=False,test_authorized=False)
    with Path(args.output).open('x',encoding='utf-8',newline='\n') as handle:
        json.dump(output,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('coverage',)},indent=2))


if __name__=='__main__': main()
