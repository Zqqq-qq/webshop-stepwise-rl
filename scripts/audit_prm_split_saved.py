"""Independent arithmetic of PR54 saved three-arm evidence, without Torch."""
import argparse
import json
import math
from pathlib import Path
import subprocess

from audit_prm_hidden_results import (PROBE_SHA, verified_archive, load_json, sha,
                                      prediction_map, score, finite)

REF='559a7791b49332375c640bc132af05e079676a83'
BUNDLE_SHA='6ffeaba47f6586e54efd3ab13fef9890d2d3bd1bad6386bbe597aa8b349bb6f5'
ROOT='results/prm-split-cosine-20261124/rtx3090-fp32-three-arms/'


def check_records(summary, labels):
    options=summary['optimization']; acc=options['gradient_accumulation_steps']
    records=summary['records']; order=list(labels); clipped=0; max_loss_delta=0.
    if acc not in (1,8) or len(records)!=summary['optimizer_steps']:
        raise ValueError('Unexpected update budget/accumulation')
    for index,row in enumerate(records):
        expected=[order[(index*acc+j)%8] for j in range(acc)]
        if (row['optimizer_step']!=index+1 or row['task_ids']!=expected or
                row['terminal_rewards']!=[labels[t] for t in expected] or
                row['samples_consumed']!=(index+1)*acc):
            raise ValueError('Update/task/label/exposure order mismatch')
        if len(row['totals_before_update'])!=acc: raise ValueError('Missing pre-update predictions')
        loss=sum((finite(p)-labels[t])**2 for t,p in zip(expected,row['totals_before_update']))/acc
        delta=abs(loss-finite(row['loss']));max_loss_delta=max(delta,max_loss_delta)
        if delta>1e-6: raise ValueError('Saved loss differs from original labels/predictions')
        norm=finite(row['gradient_norm'])
        if type(row['gradient_clipped']) is not bool or row['gradient_clipped']!=(norm>1):
            raise ValueError('Gradient clipping flag mismatch')
        clipped+=row['gradient_clipped']
        factor=(.5*(1+math.cos(math.pi*index/options['cosine_total_steps']))
                if options['lr_scheduler']=='cosine' else 1.)
        for field in ('learning_rate','head_learning_rate'):
            if not math.isclose(finite(row[field]),options[field]*factor,abs_tol=1e-18,rel_tol=1e-12):
                raise ValueError('Effective learning rate mismatch')
    if summary['samples_consumed']!=len(records)*acc or summary['samples_consumed']!=960:
        raise ValueError('Wrong total presentation budget')
    return dict(updates=len(records),presentations=summary['samples_consumed'],
                clipped_updates=clipped,max_training_loss_replay_difference=max_loss_delta)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('repo','bundle','probe-archive','output'): parser.add_argument('--'+arg,required=True)
    args=parser.parse_args()
    if Path(args.output).exists(): parser.error('Output must be new')
    files,checks=verified_archive(args.bundle,BUNDLE_SHA,
        'prm-split-cosine-rtx3090-20261124/SHA256SUMS')
    probe,_=verified_archive(args.probe_archive,PROBE_SHA)
    samples=[load_json(r) for r in probe['lrprobe/samples.jsonl'].splitlines() if r]
    labels={r['task_id']:r['terminal_reward'] for r in samples}
    # These positions are independently re-encoded by the companion tokenizer audit.
    positions={r['task_id']:r['turn_end_indices'] for r in
        load_json(probe['lrprobe/hidden-lr-1e5-probe/reference-predictions.json'])}
    reports={}; initialization=None
    for arm in ('hidden-lr-low-acc8','hidden-split-cos-acc8','hidden-split-cos-acc1'):
        def blob(name): return files[arm+'/'+name]
        summary=load_json(blob('summary.json')); manifest=load_json(blob('prm-manifest.json'))
        for name,digest in manifest['files_sha256'].items():
            if sha(blob(name))!=digest: raise ValueError('Checkpoint manifest payload SHA mismatch')
        for name in manifest['files_sha256'].keys() | {'prm-manifest.json','SHA256SUMS'}:
            if name not in ('head.pt','training-state.pt','adapter/adapter_model.safetensors'):
                tracked=subprocess.check_output(['git','show',REF+':'+ROOT+arm+'/'+name],cwd=args.repo)
                if tracked!=blob(name): raise ValueError('Tracked evidence differs from bundle')
        for name,digest in summary['identity']['source_sha256'].items():
            source=subprocess.check_output(['git','show',REF+':src/webshop_lab/prm/'+name],cwd=args.repo)
            if sha(source)!=digest: raise ValueError('Source file SHA mismatch')
        if initialization is None: initialization=summary['parameters_before']
        if initialization!=summary['parameters_before']: raise ValueError('Initial parameter digests differ')
        if manifest['identity']!=summary['identity']: raise ValueError('Summary identity differs')
        reference=prediction_map(load_json(blob('reference-predictions.json')),labels,positions)
        reload=load_json(files[arm+'-reload.json'])
        if (reload['identity']!=summary['identity'] or reload['rejected']!=[] or
                reload['reference_match'] is not True or reload['total_samples']!=8):
            raise ValueError('Saved reload metadata differs')
        restored=prediction_map(reload['predictions'],labels,positions)
        if reference!=restored: raise ValueError('Reference/reload predictions differ')
        values={t:r['total'] for t,r in reference.items()}; metrics=score(values,labels)
        full=(values[0]+values[6225])/2;zero=(values[95]+values[5031])/2
        reports[arm]=dict(metrics,**check_records(summary,labels),
            memory_gate_passed=metrics['mse']<.001 and metrics['max_absolute_error']<.1,
            full_reward_mean=full,zero_reward_mean=zero,full_minus_zero_mean=full-zero,
            focus=[dict(task_id=t,label=labels[t],total=values[t]) for t in (0,6225,95,5031)],
            saved_reload_max_difference=0.,verified_checkpoint_payloads=len(manifest['files_sha256']))
    output=dict(reviewed_commit=REF,bundle_sha256=BUNDLE_SHA,verified_bundle_payloads=len(checks),
        initial_parameter_digests_match=True,runs=reports,
        scope='Saved-output arithmetic and payload SHA only; no model execution or optimizer deserialization',
        tokenizer_crosscheck='Separate pinned-author verifier executed with original samples and real tokenizer',
        five_step_probe='Rounded text log only; no exact post-update predictions or independent reload',
        test_authorized=False)
    with Path(args.output).open('x',encoding='utf-8',newline='\n') as handle:
        json.dump(output,handle,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__': main()
