"""Frozen stdlib recomputation of report PR65 numbers; no model execution."""
import argparse
import json
import math
from pathlib import Path
import random
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from audit_prm_hidden_results import verified_archive, load_json, sha

PR65='9ce42fc5701190cf262c61fc665bfc6fa75b4045'
FULL_SHA='4ea38266297bdc5d70e1da5818851532d092c75503412a75772615b302fe797b'
GPU_SHA='85b007b1d857f47137a1c3a1bf53179713a5f8191c189c540606c4cb07e0b61c'


def paired(a,b):
    delta=[x-y for x,y in zip(a,b)]
    rng=random.Random(42)
    draws=sorted(sum(rng.choices(delta,k=len(delta)))/len(delta) for _ in range(10000))
    def quantile(p):
        pos=(len(draws)-1)*p;lo=math.floor(pos);hi=math.ceil(pos)
        return draws[lo]+(draws[hi]-draws[lo])*(pos-lo)
    return dict(difference=sum(delta)/len(delta),ci95=[quantile(.025),quantile(.975)])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('full160-archive','gpu-archive','repo','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    if Path(args.output).exists():parser.error('Output must be new')
    full,full_checks=verified_archive(args.full160_archive,FULL_SHA)
    gpu,gpu_checks=verified_archive(args.gpu_archive,GPU_SHA)
    ids=load_json(full['val_ids.json'])
    if len(set(ids))!=160 or sha(full['val_ids.json'])!='45440d724b33e69e13916d0adbe243cc63ce3bd67677b8220feff9f30b866c36':
        raise ValueError('Wrong val160 identity')
    results={};successes={}
    expected={'b0':(15,50),'b1':(54,156),'tg42':(51,155),'op42':(52,155),'tg43':(53,152),'op43':(53,151)}
    for label in expected:
        rows=[load_json(s) for s in full[label+'/rollout.jsonl'].splitlines() if s]
        meta=load_json(full[label+'/rollout.meta.json'])
        if [r['task_id'] for r in rows]!=ids or meta['selected_ids']!=ids:
            raise ValueError('Wrong task coverage/order')
        if (meta['precision']['selected_load_dtype']!='torch.float32' or
                any(r['run_id']!=meta['run_id'] for r in rows)):
            raise ValueError('Wrong precision/run identity')
        success=[int(r['done'] and r['final_reward']>=1-1e-6) for r in rows]
        bought=sum(r['done'] for r in rows)
        if (sum(success),bought)!=expected[label]:raise ValueError('Report count differs')
        results[label]=dict(tasks=160,successes=sum(success),purchases=bought,
            success_rate=sum(success)/160,
            mean_terminal_reward=sum(r['final_reward'] if r['done'] else 0. for r in rows)/160)
        successes[label]=success
    comparisons={f'{a}_minus_{b}':paired(successes[a],successes[b])
                 for a,b in [('b1','b0'),('op42','tg42'),('op43','tg43')]}
    target=load_json(gpu['target-reference.json'])['predictions'][0]
    alternative=load_json(gpu['alternative-reference.json'])['predictions'][0]
    for name,reference in [('target',target),('alternative',alternative)]:
        reload=load_json(gpu[name+'-reload.json'])
        if reload['predictions'][0]!=reference:raise ValueError('Reload reference differs')
        if reference['task_id']!=7544:raise ValueError('Wrong regression task')
    case7544=dict(task_id=7544,target_click_reward=target['step_scores'][1],
        alternative_click_reward=alternative['step_scores'][1],
        click_difference=target['step_scores'][1]-alternative['step_scores'][1],
        target_total=target['total'],alternative_total=alternative['total'],
        total_difference=target['total']-alternative['total'])
    failures={}
    for name in ('output/episodes.json','previous-four-task/arm4-episodes.json'):
        for episode in load_json(gpu[name]):
            if episode['task_id'] not in (6285,89):continue
            if episode['purchase_completed'] or episode['terminal_reward']!=0:raise ValueError('Wrong failure label')
            failures[str(episode['task_id'])]=dict(environment_terminal=0.,
                cumulative_neural_reward=sum(t['reward'] for t in episode['turns']),
                reward_turns=[t['reward'] for t in episode['turns']])
    summary=load_json(gpu['output/summary.json'])
    if summary['updates']!=1 or summary['turns']!=6:raise ValueError('Engineering scope differs')
    draft=subprocess.check_output(['git','show',PR65+':reports/final-report-experiments-draft.md'],cwd=args.repo)
    output=dict(reviewed_pr65_commit=PR65,draft_sha256=sha(draft),
        original_archive_sha256=dict(full160=FULL_SHA,pr61=GPU_SHA),
        verified_payloads=dict(full160=len(full_checks),pr61=len(gpu_checks)),
        policies=results,paired_success=comparisons,
        training_totals=load_json(full['training-completion-ledger.json'])['totals'],
        regression_7544=case7544,failed_trajectory_rewards=failures,
        engineer_record_scope=dict(updates=summary['updates'],turns=summary['turns']),
        historical_pr59_crosscheck='Prior independently accepted 31/24-pair and 48-pair records, not rerun here',
        model_or_training_executed=False,test_authorized=False)
    with Path(args.output).open('x',encoding='utf-8',newline='\n') as handle:
        json.dump(output,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
