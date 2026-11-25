"""Independent pre-execution PR55 holdout review. No rollout or training."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import zipfile

from audit_prm_hidden_results import sha, load_json, verified_archive, PROBE_SHA

REF='3fa1b9d8d546316cf237ab931df670798bd0388d'
ROOT='results/prm-expanded-fit-20261125/'


def instruction_hash(text):
    normalized=re.sub(r'^instruction\s*:\s*','',' '.join(text.lower().split())).strip()
    return sha(normalized.encode())


def isolate_candidates(pool, used, reserved, goals, interrupted):
    candidates=sorted(set(pool)-set(used))
    forbidden={instruction_hash(goals[t]) for t in set(used)|set(reserved)}
    overlap=[];duplicates=[];eligible=[];seen=set()
    for task in candidates:
        if task==interrupted: continue
        digest=instruction_hash(goals[task])
        if digest in forbidden: overlap.append(task)
        elif digest in seen: duplicates.append(task)
        else: eligible.append(task);seen.add(digest)
    selected=sorted(eligible,key=lambda t:(sha(('42:'+str(t)).encode('ascii')),t))[:160]
    return candidates,overlap,duplicates,eligible,selected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('repo','data-archive','probe-archive','goal-map','tokenizer','output'):
        parser.add_argument('--'+arg,required=True)
    args=parser.parse_args()
    if Path(args.output).exists(): parser.error('Output must be new')
    def blob(path): return subprocess.check_output(['git','show',REF+':'+path],cwd=args.repo)
    audit=load_json(blob(ROOT+'holdout-audit.json'))
    selected_blob=blob(ROOT+'holdout_ids.json');selected_recorded=load_json(selected_blob)
    splits={}
    for split,digest in audit['split_sha'].items():
        raw=blob('results/env-2026-11-17/'+split+'_ids.json')
        if sha(raw)!=digest: raise ValueError('Frozen split SHA mismatch')
        splits[split]=set(load_json(raw))
    goals_raw=Path(args.goal_map).read_bytes()
    if sha(goals_raw)!=audit['goal_map_sha256']: raise ValueError('Goal map SHA mismatch')
    goals={r['task_id']:r['instruction'] for r in load_json(goals_raw)}
    probe,_=verified_archive(args.probe_archive,PROBE_SHA)
    memory=probe['lrprobe/samples.jsonl']
    if sha(memory)!=audit['history_source_sha256']['eight_trajectory_probe']:
        raise ValueError('Probe identity mismatch')
    with zipfile.ZipFile(args.data_archive) as archive:
        manifest=load_json(archive.read('MANIFEST.sha256.json'))
        for name,digest in manifest.items():
            if sha(archive.read(name))!=digest: raise ValueError('Historical archive payload mismatch')
        def read_source(name):
            raw=archive.read('partial911-source/'+name)
            if sha(raw)!=audit['raw_input_sha256'][name]: raise ValueError('Raw source identity mismatch')
            return raw
        fit_raw=read_source('fit1451.training.jsonl')
        fit=[load_json(r) for r in fit_raw.splitlines() if r]
        dev=[load_json(r) for r in read_source('dev1451.training.jsonl').splitlines() if r]
        episodes=[load_json(r) for r in read_source('b1-train-1451.jsonl').splitlines() if r]
        metadata=load_json(read_source('b1-train-1451.meta.json'))
        histories={}
        for field,digest in audit['history_source_sha256'].items():
            if field=='eight_trajectory_probe': raw=memory
            else:
                matches=[n for n,d in manifest.items() if d==digest]
                if not matches: raise ValueError('Missing historical identity: '+field)
                raw=archive.read(matches[0])
            histories[field]=load_json(raw) if field=='previous500_dev_predictions' else [
                load_json(r) for r in raw.splitlines() if r]
    used={r['task_id'] for r in fit+dev};episode_ids=[r['task_id'] for r in episodes]
    if (len(used)!=911 or set(episode_ids)!=used or
            metadata['selected_ids'][:911]!=episode_ids):
        raise ValueError('Original rollout prefix/fit/dev identity differs')
    interrupted=metadata['selected_ids'][len(episodes)]
    if interrupted!=audit['possible_interrupted_first_unfinished']:
        raise ValueError('First unfinished task differs')
    for row in fit+dev:
        if instruction_hash(goals[row['task_id']])!=row['instruction_sha256']:
            raise ValueError('Historical instruction identity differs')
    candidates,overlap,duplicates,eligible,selected=isolate_candidates(
        splits['train'],used,splits['val']|splits['test'],goals,interrupted)
    if (overlap!=audit['excluded_instruction_overlap'] or
            duplicates!=audit['excluded_duplicate_candidate_instruction_ids'] or
            eligible!=audit['eligible_task_ids'] or selected!=selected_recorded or
            sha(selected_blob)!=audit['candidate_ids_sha256']):
        raise ValueError('Holdout exclusion/ranking/file identity mismatch')
    if {str(t):instruction_hash(goals[t]) for t in selected}!=audit['selected_instruction_sha256']:
        raise ValueError('Selected instruction hashes differ')
    history_counts={}
    for name,rows in histories.items():
        ids={r['task_id'] for r in rows}
        intersection=ids & set(candidates)
        if intersection: raise ValueError('Candidate overlaps a supplied PRM history')
        history_counts[name]=dict(count=len(ids),candidate_overlap=0)
    from transformers import AutoTokenizer
    identity=load_json(probe['lrprobe/hidden-lr-1e5-probe/summary.json'])['identity']
    for name,digest in identity['tokenizer_sha256'].items():
        if sha((Path(args.tokenizer)/name).read_bytes())!=digest: raise ValueError('Tokenizer SHA mismatch')
    tok=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True,trust_remote_code=False)
    if sha(tok.chat_template.encode())!=identity['template_sha256']: raise ValueError('Template SHA mismatch')
    kept=[];kept_lines=[]
    for raw in fit_raw.splitlines(keepends=True):
        row=load_json(raw)
        if len(tok.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=False))<=2048:
            kept.append(row);kept_lines.append(raw)
    kept_sha=sha(b''.join(kept_lines));first_six=[r['task_id'] for r in kept[:6]]
    if len(kept)!=501 or kept_sha!=audit['fit501_training_jsonl_sha256'] or first_six!=audit['fit_order_first_six_ids']:
        raise ValueError('Raw-byte retained fit501 identity differs')
    categories=Counter('max_steps_zero' if r['termination']=='max_steps' else
        ('purchase_zero' if r['terminal_reward']==0 else 'purchase_positive') for r in kept)
    output=dict(reviewed_pr55_commit=REF,scope='Pre-execution isolation of supplied PRM history only',
        split_sha256=audit['split_sha'],goal_map_sha256=sha(goals_raw),
        source_counts=dict(train=len(splits['train']),completed=len(used),fit=len(fit),historical_dev=len(dev)),
        candidate_count=len(candidates),first_unfinished_excluded=interrupted,
        excluded_instruction_overlap=overlap,excluded_internal_duplicates=duplicates,
        eligible_count=len(eligible),selected_count=len(selected),selected_ids_sha256=sha(selected_blob),
        histories_checked=history_counts,selected_task_ids=selected,
        selected_instruction_sha256={str(t):instruction_hash(goals[t]) for t in selected},
        fit501_sha256=kept_sha,retained_fit_categories=dict(categories),first_six_ids=first_six,
        planned_training=dict(updates=126,presentations=1008,first_six_presentations=3,other_presentations=2),
        limitations=['B1 SFT exposure is inherited; this is not a policy-unseen split',
                     'Undocumented device-side PRM uses still require preflight verification'],
        new_rollout_or_training_performed=False,results_pending=True,test_authorized=False)
    with Path(args.output).open('x',encoding='utf-8',newline='\n') as handle:
        json.dump(output,handle,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('selected_task_ids','selected_instruction_sha256')},indent=2))


if __name__=='__main__': main()
