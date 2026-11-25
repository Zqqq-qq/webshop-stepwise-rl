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
REF_TIMING_1127='9ace4ba8309dfbb762cdb081064e4cd7e66a80de'
REF_REFREEZE='a601999aa42f67d68fdc041f4b84a8bf5d682593'
REF_HEAD='1c19b2bb0c3f3e34856bad134f8699247d2b9cde'
ROOT='results/prm-expanded-fit-20261125/'
PR55_OUTSIDE_AUDIT_SHA='edfe3ca0a4c9cc7943810dad3c255d8cfc56b91db06df2ea6c9a58d8988d5d13'
PR55_UNFILTERED_ZIP_SHA='0df7049d0f4788cae5f6277b9ca900feea5bed6f71b435718ca3fd27d2f86a15'
PR55_REFREEZE_HOLDOUT_AUDIT_SHA='17abdcc658310d620902d53f3fb403c8464457b2967f1fef5cfa561ca588334a'
PR55_ANNOUNCED_ORIG160_27=(
    6472,7033,7275,7284,7290,7553,7719,8015,8081,8305,8372,8586,8672,8757,
    8946,9014,9142,9228,9249,9336,9672,9807,9818,10050,10589,10644,11053)
PR55_ANNOUNCED_EXPOSED_85=(
    6472,6504,6632,6662,6670,6714,6752,6935,7015,7033,7070,7219,7275,7283,
    7284,7290,7307,7362,7369,7553,7600,7636,7719,7730,7750,7793,8015,8028,
    8081,8086,8118,8159,8170,8264,8305,8332,8372,8526,8546,8586,8618,8672,
    8730,8732,8742,8757,8810,8925,8946,8980,9014,9067,9142,9202,9228,9240,
    9249,9258,9284,9336,9479,9672,9674,9700,9754,9807,9815,9818,9860,9980,
    10050,10409,10518,10589,10596,10644,10730,10756,10822,10921,10970,11027,
    11053,11130,11333)
PR55_REPLACEMENT_CANDIDATE_SHA='15b96f015ed8a2b804633523144011eb2fdf092903ccded17395020b925eb5c1'
PR55_HANDOFF_ZIP_SHA='a6034a1b192e834d492d52c1d32d82973a5e561250771e2407941f6924657022'
PR55_DECLARED_HANDOFF_SHAS={
    'preflight-summary.json':'c1ed10885a1d0b5c60eb2e7f302ff28cd5282f008fb708849c7f587261fa894f',
    'full-history-audit.json':'a3cad07fd70d7bb1183ac71e5f1a922c720f4a1106785bef348c8e3703ee4d2a',
    'eligible-history-audit.json':'66a9d5d0542d6cf21976a00c7a6892f622e504098d82c2ff439e5048aa0278ee',
    'holdout-replacement-candidate.json':PR55_REPLACEMENT_CANDIDATE_SHA,
}
PLAN_ONLY_KEYS={'selected_ids','train_ids','val_ids','test_ids','eligible_task_ids'}


def instruction_hash(text):
    normalized=re.sub(r'^instruction\s*:\s*','',' '.join(text.lower().split())).strip()
    return sha(normalized.encode())


def common_instruction_hash(text):
    normalized=' '.join(text.lower().split()).rstrip('.')
    normalized=re.sub(r',?\s*and price lower than [\d.]+ dollars\.?$','',normalized).strip().rstrip('.')
    return sha(normalized.encode())


def canonical_ids_bytes(ids):
    return (json.dumps(list(ids),indent=2)+'\n').encode('utf-8')


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


def select_replacement_candidates(eligible, orig_selected, exposed_eligible_ids, goals,
                                  all_exposed_ids=None, count=160):
    eligible_list=list(eligible);eligible_set=set(eligible_list)
    if len(eligible_set)!=len(eligible_list):
        raise ValueError('Eligible task list contains duplicates')
    exposed_list=list(exposed_eligible_ids);exposed_set=set(exposed_list)
    if len(exposed_set)!=len(exposed_list):
        raise ValueError('Exposed eligible task list contains duplicates')
    unknown=sorted(exposed_set-eligible_set)
    if unknown:
        raise ValueError('Exposed IDs outside eligible set: '+str(unknown[:5]))
    remaining_by_id=sorted(eligible_set-exposed_set)
    extra_exposed=set(exposed_set)|(set(all_exposed_ids) if all_exposed_ids is not None else set())
    exposed_hashes={instruction_hash(goals[t]) for t in extra_exposed if t in goals}
    excluded_by_twin=[];remaining=[]
    for task in remaining_by_id:
        if instruction_hash(goals[task]) in exposed_hashes:
            excluded_by_twin.append(task)
        else:
            remaining.append(task)
    if len(remaining)<count:
        raise ValueError('Not enough remaining candidates: '+str(len(remaining)))
    selected=sorted(remaining,key=lambda t:(sha(('42:'+str(t)).encode('ascii')),t))[:count]
    selected_set=set(selected);orig_list=list(orig_selected);orig_set=set(orig_list)
    retained=[t for t in selected if t in orig_set]
    expected_retained=[t for t in orig_list if t in selected_set]
    if retained!=expected_retained:
        raise ValueError('Relative order of retained holdout IDs changed')
    removed=[t for t in orig_list if t not in selected_set]
    added=[t for t in selected if t not in orig_set]
    selected_hashes={str(t):instruction_hash(goals[t]) for t in selected}
    if len(set(selected_hashes.values()))!=len(selected):
        raise ValueError('Selected replacement IDs contain duplicate instructions')
    by_hash={}
    for tid,text in goals.items():
        by_hash.setdefault(instruction_hash(text),[]).append(tid)
    twins={}
    for task in remaining_by_id:
        others=sorted(x for x in by_hash[instruction_hash(goals[task])] if x!=task)
        if others: twins[str(task)]=others
    blob=canonical_ids_bytes(selected)
    return dict(
        eligible_before=len(eligible_list),
        exposed_eligible_count=len(exposed_set),
        remaining_after_id_exclusion_count=len(remaining_by_id),
        excluded_by_instruction_twin=excluded_by_twin,
        remaining_after_instruction_exclusion_count=len(remaining),
        selected_count=len(selected),
        retained_count=len(retained),
        removed_from_original_160=sorted(removed),
        added_replacement_ids=added,
        selected_task_ids=selected,
        selected_ids_sha256=sha(blob),
        selected_instruction_sha256=selected_hashes,
        cross_goal_instruction_twins_in_remaining=twins,
        cross_goal_instruction_twins_in_selected={
            str(t):twins[str(t)] for t in selected if str(t) in twins},
    )


def extract_actual_task_ids(obj, found=None):
    if found is None: found=set()
    if isinstance(obj,dict):
        tid=obj.get('task_id')
        if type(tid) is int: found.add(tid)
        for key,value in obj.items():
            if key in PLAN_ONLY_KEYS: continue
            if key in ('task_ids','fit_task_ids','dev_task_ids','kept_task_ids',
                       'rejected_task_ids','exposed_task_ids','observed_task_ids') and isinstance(value,list):
                for item in value:
                    if type(item) is int: found.add(item)
            elif key=='sample_keys' and isinstance(value,list):
                for item in value:
                    if isinstance(item,list) and item and type(item[0]) is int: found.add(item[0])
            elif key in ('by_task_id','task_records','exposed_by_task','task_exposures','tasks') and isinstance(value,dict):
                for k,v in value.items():
                    if isinstance(k,str) and k.isdigit(): found.add(int(k))
                    extract_actual_task_ids(v,found)
            else:
                extract_actual_task_ids(value,found)
    elif isinstance(obj,list):
        for item in obj: extract_actual_task_ids(item,found)
    return found


def _scan_json_or_jsonl_bytes(name, raw):
    found=set()
    if name.endswith('.json'):
        try: extract_actual_task_ids(json.loads(raw),found)
        except Exception: pass
    elif name.endswith('.jsonl'):
        for line in raw.splitlines():
            if not line.strip(): continue
            try: extract_actual_task_ids(json.loads(line),found)
            except Exception: pass
    return found


def scan_published_sources(repo, git_refs, zip_paths, target_ids):
    targets=set(target_ids);hits={};sources=[]
    for ref in git_refs:
        commit=subprocess.check_output(['git','rev-parse',ref],cwd=repo).decode().strip()
        files=subprocess.check_output(['git','ls-tree','-r','--name-only',commit,'results/'],cwd=repo).decode().splitlines()
        ref_hits=set()
        for rel in files:
            if not rel.endswith(('.json','.jsonl')): continue
            raw=subprocess.check_output(['git','show',commit+':'+rel],cwd=repo)
            matched=_scan_json_or_jsonl_bytes(rel,raw) & targets
            for tid in sorted(matched):
                hits.setdefault(tid,[]).append(ref+':'+rel)
                ref_hits.add(tid)
        sources.append(dict(kind='git_ref',ref=ref,commit=commit,matched_ids=sorted(ref_hits)))
    for path_str in zip_paths:
        path=Path(path_str);digest=sha(path.read_bytes());zip_hits=set()
        with zipfile.ZipFile(path) as archive:
            for member in sorted(archive.namelist()):
                if not member.endswith(('.json','.jsonl')): continue
                matched=_scan_json_or_jsonl_bytes(member,archive.read(member)) & targets
                for tid in sorted(matched):
                    hits.setdefault(tid,[]).append(path.name+'::'+member)
                    zip_hits.add(tid)
        sources.append(dict(kind='zip',name=path.name,sha256=digest,matched_ids=sorted(zip_hits)))
    return sources,{str(k):sorted(set(v)) for k,v in sorted(hits.items())}


def _assert_zero_skipped(obj, path=''):
    if isinstance(obj,dict):
        for key,value in obj.items():
            if key in ('skipped','parse_skipped','skipped_files','unparsed_files') and isinstance(value,int) and value!=0:
                raise ValueError('Non-zero skipped audit count at '+path+'.'+key+': '+str(value))
            _assert_zero_skipped(value,path+'.'+key)
    elif isinstance(obj,list):
        for idx,item in enumerate(obj): _assert_zero_skipped(item,path+'['+str(idx)+']')


def verify_device_handoff(handoff_dir, expected_shas, eligible, orig_selected, goals, count=160):
    root=Path(handoff_dir);verified_shas={};payloads={};raw_bytes={}
    for name,expected in expected_shas.items():
        path=root/name
        if not path.is_file(): raise ValueError('Missing handoff file: '+name)
        raw=path.read_bytes();digest=sha(raw)
        if expected and digest!=expected:
            raise ValueError('Handoff SHA mismatch for '+name+': '+digest)
        verified_shas[name]=digest;raw_bytes[name]=raw;payloads[name]=load_json(raw)
    _assert_zero_skipped(payloads['preflight-summary.json'],'preflight-summary')
    _assert_zero_skipped(payloads['full-history-audit.json'],'full-history-audit')
    _assert_zero_skipped(payloads['eligible-history-audit.json'],'eligible-history-audit')
    eligible_set=set(eligible)
    elig_exposed=extract_actual_task_ids(payloads['eligible-history-audit.json']) & eligible_set
    full_exposed=extract_actual_task_ids(payloads['full-history-audit.json'])
    if not elig_exposed:
        raise ValueError('No exposed eligible task IDs parsed from eligible-history-audit.json')
    if not elig_exposed.issubset(full_exposed):
        raise ValueError('eligible-history-audit exposed IDs not covered by full-history-audit')
    result=select_replacement_candidates(
        eligible,orig_selected,sorted(elig_exposed),goals,all_exposed_ids=full_exposed,count=count)
    candidate_ids=payloads['holdout-replacement-candidate.json']
    if candidate_ids!=result['selected_task_ids']:
        raise ValueError('Replacement candidate IDs disagree with ID+instruction-isolated recomputation')
    if raw_bytes['holdout-replacement-candidate.json']!=canonical_ids_bytes(result['selected_task_ids']):
        raise ValueError('Replacement candidate raw JSON bytes differ from canonical serialization')
    result['verified_handoff_shas']=verified_shas
    result['full_exposed_count']=len(full_exposed)
    result['device_handoff_received']=True
    return result


def _run_replacement_preflight(args):
    def blob(path): return subprocess.check_output(['git','show',REF+':'+path],cwd=args.repo)
    audit=load_json(blob(ROOT+'holdout-audit.json'))
    orig_blob=blob(ROOT+'holdout_ids.json');orig_selected=load_json(orig_blob)
    if sha(orig_blob)!=audit['candidate_ids_sha256']:
        raise ValueError('Frozen PR55 holdout_ids SHA mismatch')
    splits={}
    for split,digest in audit['split_sha'].items():
        raw=blob('results/env-2026-11-17/'+split+'_ids.json')
        if sha(raw)!=digest: raise ValueError('Frozen split SHA mismatch')
        splits[split]=set(load_json(raw))
    goals_raw=Path(args.goal_map).read_bytes()
    if sha(goals_raw)!=audit['goal_map_sha256']: raise ValueError('Goal map SHA mismatch')
    goals={r['task_id']:r['instruction'] for r in load_json(goals_raw)}
    eligible=audit['eligible_task_ids']
    interrupted=audit['possible_interrupted_first_unfinished']
    excluded_overlap=set(audit['excluded_instruction_overlap'])
    excluded_dups=set(audit['excluded_duplicate_candidate_instruction_ids'])
    used_911=splits['train']-set(eligible)-{interrupted}-excluded_overlap-excluded_dups
    if len(used_911)!=911: raise ValueError('Historical 911 completed set mismatch')
    announced_85=list(PR55_ANNOUNCED_EXPOSED_85)
    orig_overlap=sorted(set(announced_85)&set(orig_selected))
    if orig_overlap!=sorted(PR55_ANNOUNCED_ORIG160_27):
        raise ValueError('Announced 27 original-160 overlap mismatch')
    repl=select_replacement_candidates(eligible,orig_selected,announced_85,goals,count=160)
    if repl['selected_ids_sha256']!=PR55_REPLACEMENT_CANDIDATE_SHA:
        raise ValueError('Replacement candidate SHA mismatch: '+repl['selected_ids_sha256'])
    rem_445=sorted(set(eligible)-set(announced_85))
    inst_checks={}
    for label,id_set in (
        ('used_911',used_911),('val_160',splits['val']),('test_200',splits['test']),
        ('announced_exposed_85',set(announced_85)),('interrupted_6426',{interrupted}),
        ('excluded_overlap_6',excluded_overlap),('excluded_internal_duplicates_3',excluded_dups)):
        hset={instruction_hash(goals[t]) for t in id_set}
        inst_checks[label]=dict(
            overlap_in_remaining_445=[t for t in rem_445 if instruction_hash(goals[t]) in hset],
            overlap_in_replacement_160=[t for t in repl['selected_task_ids'] if instruction_hash(goals[t]) in hset],
        )
    twin_ids={x for xs in repl['cross_goal_instruction_twins_in_remaining'].values() for x in xs}
    sources,hits=scan_published_sources(args.repo,args.git_ref,args.archive,set(eligible)|twin_ids)
    pub_in_530=sorted(int(k) for k in hits if int(k) in set(eligible))
    pub_in_85=sorted(t for t in pub_in_530 if t in set(announced_85))
    pub_in_orig160=sorted(t for t in pub_in_530 if t in set(orig_selected))
    pub_in_445=sorted(t for t in pub_in_530 if t in set(rem_445))
    pub_twins=sorted(int(k) for k in hits if int(k) in twin_ids)
    clean_with_pub_twins=select_replacement_candidates(
        eligible,orig_selected,announced_85,goals,
        all_exposed_ids=set(announced_85)|set(pub_twins),count=160)
    output=dict(
        reviewed_pr55_commit=REF,
        mode='replacement-preflight',
        scope='Pre-handoff arithmetic, instruction-isolation and published-artifact audit of How announced 85 exposed IDs; awaiting 4 device audit files',
        split_sha256=audit['split_sha'],
        goal_map_sha256=sha(goals_raw),
        original_holdout_ids_sha256=audit['candidate_ids_sha256'],
        announced_exposed_85_count=len(announced_85),
        announced_orig160_overlap_27=orig_overlap,
        replacement_recomputation=repl,
        instruction_isolation_checks=inst_checks,
        published_artifact_crosscheck=dict(
            sources=sources,
            confirmed_in_eligible_530=pub_in_530,
            confirmed_in_announced_85=pub_in_85,
            confirmed_in_original_160=pub_in_orig160,
            confirmed_in_remaining_445=pub_in_445,
            confirmed_instruction_twins_outside_530=pub_twins,
            hits_by_task_id=hits,
            remaining_after_public_twin_exclusion_count=clean_with_pub_twins['remaining_after_instruction_exclusion_count'],
            excluded_by_public_instruction_twin=clean_with_pub_twins['excluded_by_instruction_twin'],
            replacement_160_unchanged_after_public_twin_exclusion=(
                clean_with_pub_twins['selected_ids_sha256']==PR55_REPLACEMENT_CANDIDATE_SHA),
        ),
        declared_handoff_files_pending=PR55_DECLARED_HANDOFF_SHAS,
        device_handoff_received=False,
        re_freeze_authorized=False,
        new_rollout_or_training_performed=False,
        test_authorized=False,
    )
    with Path(args.output).open('x',encoding='utf-8',newline='\n') as handle:
        json.dump(output,handle,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps({
        'mode':output['mode'],'replacement_sha256':repl['selected_ids_sha256'],
        'retained':repl['retained_count'],'added':len(repl['added_replacement_ids']),
        'published_confirmed_85':len(pub_in_85),'published_confirmed_orig160':len(pub_in_orig160),
        'published_confirmed_445':len(pub_in_445),'published_confirmed_twins':pub_twins,
        'excluded_by_public_twin':clean_with_pub_twins['excluded_by_instruction_twin'],
        'device_handoff_received':False,'re_freeze_authorized':False},indent=2))


def _audit_history_doc(doc, label):
    if doc.get('skipped_json_files'):
        raise ValueError(label+' has non-empty skipped_json_files')
    fbc=doc['files_by_class']
    re_class={k:sorted(set().union(*(set(v) for v in files.values()))) if files else []
              for k,files in fbc.items()}
    re_all=sorted(set().union(*(set(v) for v in re_class.values()))) if re_class else []
    for key,cls in (('prm_unique_ids','prm'),('rl_unique_ids','rl'),('other_unique_ids','other')):
        if re_class[cls]!=doc[key]: raise ValueError(label+' class union mismatch: '+cls)
    if re_all!=doc['all_history_unique_ids'] or len(re_all)!=doc['all_history_unique_count']:
        raise ValueError(label+' total union mismatch')
    return re_class,re_all


def _run_replacement_handoff(args):
    def blob(path): return subprocess.check_output(['git','show',REF+':'+path],cwd=args.repo)
    audit=load_json(blob(ROOT+'holdout-audit.json'))
    orig_blob=blob(ROOT+'holdout_ids.json');orig_selected=load_json(orig_blob)
    frozen_cfg=load_json(blob(ROOT+'dev-rollout.json'))
    if sha(orig_blob)!=audit['candidate_ids_sha256']:
        raise ValueError('Frozen PR55 holdout_ids SHA mismatch')
    splits={}
    for split,digest in audit['split_sha'].items():
        raw=blob('results/env-2026-11-17/'+split+'_ids.json')
        if sha(raw)!=digest: raise ValueError('Frozen split SHA mismatch')
        splits[split]=set(load_json(raw))
    goals_raw=Path(args.goal_map).read_bytes()
    if sha(goals_raw)!=audit['goal_map_sha256']: raise ValueError('Goal map SHA mismatch')
    goals={r['task_id']:r['instruction'] for r in load_json(goals_raw)}
    if args.handoff_zip:
        zpath=Path(args.handoff_zip);zbytes=zpath.read_bytes();zsha=sha(zbytes)
        if zsha!=PR55_HANDOFF_ZIP_SHA: raise ValueError('Handoff ZIP SHA mismatch: '+zsha)
        members={}
        with zipfile.ZipFile(zpath) as zf:
            for name in zf.namelist():
                if not name.endswith('/'): members[name]=zf.read(name)
        sums_lines=members['SHA256SUMS'].decode('utf-8').strip().splitlines()
        verified_sums={}
        for line in sums_lines:
            expected,fname=line.split()
            actual=sha(members[fname])
            if actual!=expected: raise ValueError('SHA256SUMS mismatch: '+fname)
            verified_sums[fname]=actual
        for name,expected in PR55_DECLARED_HANDOFF_SHAS.items():
            if verified_sums.get(name)!=expected: raise ValueError('Declared handoff SHA mismatch: '+name)
        summary=load_json(members['preflight-summary.json'])
        full_audit=load_json(members['full-history-audit.json'])
        elig_audit=load_json(members['eligible-history-audit.json'])
        cand_raw=members['holdout-replacement-candidate.json']
        cand_ids=load_json(cand_raw)
        local_cfg=load_json(members['dev-rollout.local.json'])
        model_pre=load_json(members['model-load-preflight.json'])
        full_script=members['full_history_audit.py'].decode('utf-8')
        elig_script=members['eligible_history_audit.py'].decode('utf-8')
        for key,rel in (
            ('holdout_ids_sha256',ROOT+'holdout_ids.json'),
            ('holdout_audit_sha256',ROOT+'holdout-audit.json'),
            ('dev_rollout_sha256',ROOT+'dev-rollout.json'),
            ('b1_identity_sha256','configs/identities/b1-pr18.json')):
            if sha(blob(rel))!=summary['frozen_inputs'][key]:
                raise ValueError('Preflight frozen_inputs SHA mismatch: '+key)
        cfg_diffs={k:dict(frozen=frozen_cfg.get(k),local=local_cfg.get(k))
                   for k in sorted(set(frozen_cfg)|set(local_cfg))
                   if frozen_cfg.get(k)!=local_cfg.get(k)}
        full_cls,full_all=_audit_history_doc(full_audit,'full-history-audit')
        elig_cls,elig_all=_audit_history_doc(elig_audit,'eligible-history-audit')
        if set(full_all)!=(set(elig_all)&set(orig_selected)):
            raise ValueError('full-history-audit is not exact restriction of eligible-history-audit to orig_160')
        if summary['history_checks']['eligible_pool_exposed_ids']!=elig_all:
            raise ValueError('Summary eligible_pool_exposed_ids differs from eligible-history-audit')
        if summary['history_checks']['union_unique_ids']!=full_all:
            raise ValueError('Summary union_unique_ids differs from full-history-audit')
        # Separate genuine PRM/RL/progress-estimator records from sft-label-length-audit.json copies
        id_files={}
        for cls,files in elig_audit['files_by_class'].items():
            for fpath,ids in files.items():
                for tid in ids: id_files.setdefault(tid,[]).append((cls,fpath))
        sft_audit_only=sorted(t for t,es in id_files.items()
                              if all('sft-label-length-audit.json' in p for _,p in es))
        real_exposed=sorted(set(elig_all)-set(sft_audit_only))
        eligible=audit['eligible_task_ids']
        repl=select_replacement_candidates(eligible,orig_selected,elig_all,goals,count=160)
        if cand_ids!=repl['selected_task_ids'] or cand_raw!=canonical_ids_bytes(repl['selected_task_ids']):
            raise ValueError('holdout-replacement-candidate.json disagrees with recomputed replacement list')
        repl_real=select_replacement_candidates(eligible,orig_selected,real_exposed,goals,count=160)
        # Check whether the device scanner filtered to target sets (blocking external-ID instruction check)
        full_filtered_to_holdout='task_id in holdout' in full_script
        elig_filtered_to_targets='task_id in targets' in elig_script
        twin_ids={x for xs in repl['cross_goal_instruction_twins_in_remaining'].values() for x in xs}
        sources,hits=scan_published_sources(args.repo,args.git_ref,args.archive,set(eligible)|twin_ids)
        pub_in_530=sorted(int(k) for k in hits if int(k) in set(eligible))
        pub_in_85=sorted(t for t in pub_in_530 if t in set(elig_all))
        pub_in_orig160=sorted(t for t in pub_in_530 if t in set(orig_selected))
        pub_in_445=sorted(t for t in pub_in_530 if t not in set(elig_all))
        pub_twins=sorted(int(k) for k in hits if int(k) in twin_ids)
        clean_pub=select_replacement_candidates(
            eligible,orig_selected,elig_all,goals,all_exposed_ids=set(elig_all)|set(pub_twins),count=160)
        output=dict(
            reviewed_pr55_commit=REF,
            mode='replacement-handoff-zip',
            handoff_zip=dict(name=zpath.name,size_bytes=len(zbytes),sha256=zsha,
                             verified_members=len(verified_sums),sha256sums=verified_sums),
            preflight_checks=dict(
                status=summary['status'],
                collection_authorized=summary['collection_authorized'],
                training_authorized=summary['training_authorized'],
                rollout_executed=summary['rollout_executed'],
                training_executed=summary['training_executed'],
                frozen_holdout_modified=summary['frozen_holdout_modified'],
                local_rollout_config_diffs=cfg_diffs,
                model_load_status=model_pre['status'],
                generation_config_file_sha256=model_pre['generation_config_file_sha256'],
                effective_generation=summary['device_checks']['effective_generation'],
                environment_goal_fingerprint=summary['environment']['health']['goal_fingerprint'],
                environment_alignment_mismatches=summary['environment']['health']['alignment_mismatches'],
            ),
            history_audit_verification=dict(
                scanned_json_files=elig_audit['scanned_json_files'],
                skipped_json_files=len(elig_audit['skipped_json_files']),
                orig160_exposed_count=len(full_all),
                eligible530_exposed_count=len(elig_all),
                eligible530_class_counts={k:len(v) for k,v in elig_cls.items()},
                sft_label_length_audit_only_ids=sft_audit_only,
                real_non_sft_exposed_count=len(real_exposed),
                sft_audit_only_affects_top160=(repl_real['selected_ids_sha256']!=repl['selected_ids_sha256']),
                scanner_filtered_full_to_holdout160=full_filtered_to_holdout,
                scanner_filtered_eligible_to_530=elig_filtered_to_targets,
                external_id_instruction_isolation_verified_by_handoff=(
                    not full_filtered_to_holdout and not elig_filtered_to_targets),
            ),
            replacement_recomputation=repl,
            published_artifact_crosscheck=dict(
                sources=sources,
                confirmed_in_85=pub_in_85,
                confirmed_in_original_160=pub_in_orig160,
                confirmed_in_remaining_445=pub_in_445,
                confirmed_external_instruction_twins=pub_twins,
                hits_by_task_id=hits,
                excluded_by_public_instruction_twin=clean_pub['excluded_by_instruction_twin'],
                remaining_after_public_twin_exclusion_count=clean_pub['remaining_after_instruction_exclusion_count'],
                replacement_160_unchanged_after_public_twin_exclusion=(
                    clean_pub['selected_ids_sha256']==PR55_REPLACEMENT_CANDIDATE_SHA),
            ),
            required_device_twin_ids_to_check=sorted(twin_ids),
            required_selected160_twin_ids_to_check=sorted(
                x for xs in repl['cross_goal_instruction_twins_in_selected'].values() for x in xs),
            device_handoff_received=True,
            re_freeze_authorized=False,
            re_freeze_blocker='Device scanner filtered task_id to 160/530 targets and omitted external IDs; twin 7900->6755 is already confirmed in PR46 dataset-200-manifest.json, and external twins 8000->8547, 2525->8778 (in candidate 160) plus 9381/11798/6045 must be checked on device before formal re-freeze.',
            new_rollout_or_training_performed=False,
            test_authorized=False,
        )
        with Path(args.output).open('x',encoding='utf-8',newline='\n') as handle:
            json.dump(output,handle,indent=2,allow_nan=False);handle.write('\n')
        print(json.dumps({
            'mode':output['mode'],'zip_sha256':zsha,'verified_members':len(verified_sums),
            'exposed_85':len(elig_all),'sft_audit_only':sft_audit_only,'real_exposed':len(real_exposed),
            'replacement_sha256':repl['selected_ids_sha256'],'pub_confirmed_85':len(pub_in_85),
            'pub_twins':pub_twins,'excluded_by_pub_twin':clean_pub['excluded_by_instruction_twin'],
            'required_selected160_twins':output['required_selected160_twin_ids_to_check'],
            're_freeze_authorized':False},indent=2))
        return
    result=verify_device_handoff(
        args.handoff_dir,PR55_DECLARED_HANDOFF_SHAS,audit['eligible_task_ids'],orig_selected,goals,count=160)
    output=dict(reviewed_pr55_commit=REF,mode='replacement-handoff',
        goal_map_sha256=sha(goals_raw),result=result,
        new_rollout_or_training_performed=False,test_authorized=False)
    with Path(args.output).open('x',encoding='utf-8',newline='\n') as handle:
        json.dump(output,handle,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps({k:v for k,v in result.items()
                      if k not in ('selected_task_ids','selected_instruction_sha256')},indent=2))


def verify_outside_candidate_audit(path, expected_sha=None, expected_targets=(2525,8000)):
    p=Path(path);raw=p.read_bytes();digest=sha(raw)
    if expected_sha and digest!=expected_sha:
        raise ValueError('Outside-candidate audit SHA mismatch: '+digest)
    doc=load_json(raw)
    if doc.get('skipped_json_files'):
        raise ValueError('Outside-candidate audit has non-empty skipped_json_files')
    if not isinstance(doc.get('scanned_json_files'),int) or doc['scanned_json_files']<=0:
        raise ValueError('Outside-candidate audit has invalid scanned_json_files')
    if sorted(doc.get('target_task_ids',[]))!=sorted(expected_targets):
        raise ValueError('Outside-candidate audit target_task_ids mismatch')
    if doc.get('hit_task_ids') or doc.get('hit_file_count',0)!=0 or doc.get('files'):
        raise ValueError('Outside-candidate audit reported unexpected hits: '+str(doc.get('hit_task_ids')))
    return dict(
        name=p.name,size_bytes=len(raw),sha256=digest,
        target_task_ids=doc['target_task_ids'],
        scope=doc.get('scope',''),
        roots=doc.get('roots',[]),
        scanned_json_files=doc['scanned_json_files'],
        skipped_json_files_count=len(doc.get('skipped_json_files',[])),
        hit_task_ids=doc['hit_task_ids'],
        hit_file_count=doc['hit_file_count'],
        task_id_only_scan=True,
    )


def verify_unfiltered_history_zip(zip_path, goals, expected_twins_map, expected_zip_sha=None):
    zpath=Path(zip_path);zbytes=zpath.read_bytes();zsha=sha(zbytes)
    if expected_zip_sha and zsha!=expected_zip_sha:
        raise ValueError('Unfiltered history ZIP SHA mismatch: '+zsha)
    members={}
    with zipfile.ZipFile(zpath) as zf:
        for name in zf.namelist():
            if not name.endswith('/'): members[name]=zf.read(name)
    if 'UNFILTERED_SHA256SUMS' not in members:
        raise ValueError('Missing UNFILTERED_SHA256SUMS in unfiltered audit ZIP')
    verified_sums={}
    for line in members['UNFILTERED_SHA256SUMS'].decode('utf-8').strip().splitlines():
        expected,fname=line.split()
        if fname not in members: raise ValueError('Missing member listed in UNFILTERED_SHA256SUMS: '+fname)
        actual=sha(members[fname])
        if actual!=expected: raise ValueError('UNFILTERED_SHA256SUMS mismatch: '+fname)
        verified_sums[fname]=actual
    doc=load_json(members['unfiltered-history-audit.json'])
    if doc.get('skipped_json_files'):
        raise ValueError('Unfiltered audit has non-empty skipped_json_files')
    if not isinstance(doc.get('scanned_json_files'),int) or doc['scanned_json_files']<=0:
        raise ValueError('Unfiltered audit has invalid scanned_json_files')
    scope=doc['scope']
    if scope.get('candidate_filter_applied') is not False:
        raise ValueError('Unfiltered audit candidate_filter_applied must be false')
    required_fields={'task_id','task_ids','sample_keys','sample_key'}
    if not required_fields.issubset(set(scope.get('actual_use_fields',[]))):
        raise ValueError('Unfiltered audit missing required actual_use_fields')
    if 'selected_ids' not in scope.get('excluded_plan_fields',[]):
        raise ValueError('Unfiltered audit must exclude selected_ids plan field')
    exp_pairs={str(k):v for k,v in sorted(expected_twins_map.items())}
    if doc.get('candidate_pairs')!=exp_pairs or sorted(doc.get('target_task_ids',[]))!=sorted(expected_twins_map.keys()):
        raise ValueError('Unfiltered audit targets/candidate_pairs disagree with goal-map twins')
    inst_checks=doc.get('instruction_checks',{})
    for tid,cid in sorted(expected_twins_map.items()):
        rec=inst_checks.get(str(tid))
        if not rec: raise ValueError('Missing instruction_check for '+str(tid))
        th=common_instruction_hash(goals[tid]);ch=common_instruction_hash(goals[cid])
        if (rec.get('candidate_id')!=cid or rec.get('target_instruction_sha256')!=th or
                rec.get('candidate_instruction_sha256')!=ch or th!=ch or
                instruction_hash(goals[tid])!=instruction_hash(goals[cid]) or
                rec.get('same_normalized_instruction') is not True):
            raise ValueError('Instruction hash verification failed for twin '+str(tid)+'->'+str(cid))
    actual_matches=doc.get('actual_matches',[])
    re_hits=sorted({m['task_id'] for m in actual_matches})
    if re_hits!=doc.get('actual_hit_ids') or len(re_hits)!=doc.get('actual_hit_count'):
        raise ValueError('Unfiltered audit actual_hit_ids mismatch with actual_matches')
    nonuse_matches=doc.get('nonuse_matches',[])
    nonuse_files=sorted({m['file'] for m in nonuse_matches})
    zero_hit=sorted(set(expected_twins_map.keys())-set(re_hits))
    return dict(
        name=zpath.name,size_bytes=len(zbytes),sha256=zsha,
        verified_members=len(verified_sums),sha256sums=verified_sums,
        scanned_json_files=doc['scanned_json_files'],
        skipped_json_files_count=len(doc.get('skipped_json_files',[])),
        roots=scope.get('roots',[]),
        actual_use_fields=scope['actual_use_fields'],
        excluded_plan_fields=scope['excluded_plan_fields'],
        candidate_filter_applied=scope['candidate_filter_applied'],
        field_occurrence_counts=doc.get('field_occurrence_counts',{}),
        target_task_ids=doc['target_task_ids'],
        candidate_pairs=doc['candidate_pairs'],
        actual_hit_ids=re_hits,
        zero_hit_target_ids=zero_hit,
        actual_match_count=len(actual_matches),
        actual_matches=[dict(task_id=m['task_id'],field=m['field'],json_path=m['json_path'],
                             file=m['file'],size_bytes=m['size_bytes'],sha256=m['sha256'])
                        for m in actual_matches],
        nonuse_match_count=len(nonuse_matches),
        nonuse_unique_files=nonuse_files,
        replacement_impact=doc.get('replacement_impact',{}),
        instruction_checks_verified=True,
        goal_map_sha256=doc.get('goal_map_sha256'),
    )


def _run_refreeze_audit(args):
    def blob(ref, path):
        return subprocess.check_output(['git','show',ref+':'+path],cwd=args.repo)
    head_ref=args.refreeze_ref
    for anc,desc in ((REF,REF_TIMING_1127),(REF_TIMING_1127,REF_REFREEZE),(REF_REFREEZE,head_ref)):
        subprocess.check_call(['git','merge-base','--is-ancestor',anc,desc],cwd=args.repo)
    orig_audit=load_json(blob(REF,ROOT+'holdout-audit.json'))
    orig_ids_blob=blob(REF,ROOT+'holdout_ids.json');orig_selected=load_json(orig_ids_blob)
    orig_cfg_blob=blob(REF,ROOT+'dev-rollout.json');orig_cfg=load_json(orig_cfg_blob)
    refreeze_audit_blob=blob(REF_REFREEZE,ROOT+'holdout-audit.json')
    head_audit_blob=blob(head_ref,ROOT+'holdout-audit.json')
    if refreeze_audit_blob!=head_audit_blob:
        raise ValueError('holdout-audit.json changed between refreeze commit and head_ref')
    if sha(head_audit_blob)!=PR55_REFREEZE_HOLDOUT_AUDIT_SHA:
        raise ValueError('Refreeze holdout-audit.json SHA mismatch: '+sha(head_audit_blob))
    refreeze_ids_blob=blob(REF_REFREEZE,ROOT+'holdout_ids.json')
    head_ids_blob=blob(head_ref,ROOT+'holdout_ids.json')
    if refreeze_ids_blob!=head_ids_blob:
        raise ValueError('holdout_ids.json changed between refreeze commit and head_ref')
    if blob(head_ref,ROOT+'dev-rollout.json')!=orig_cfg_blob:
        raise ValueError('dev-rollout.json unexpectedly changed after original commit')
    new_audit=load_json(head_audit_blob);new_selected=load_json(head_ids_blob)
    goals_raw=Path(args.goal_map).read_bytes();goals_sha=sha(goals_raw)
    if goals_sha!=orig_audit['goal_map_sha256'] or goals_sha!=new_audit['goal_map_sha256']:
        raise ValueError('Goal map SHA mismatch')
    goals={r['task_id']:r['instruction'] for r in load_json(goals_raw)}
    # Verify unchanged frozen inputs/splits/fit501 identity
    for field in ('split_sha','raw_input_sha256','history_source_sha256',
                  'fit501_training_jsonl_sha256','fit_order_first_six_ids',
                  'possible_interrupted_first_unfinished','excluded_instruction_overlap',
                  'excluded_duplicate_candidate_instruction_ids'):
        if orig_audit[field]!=new_audit[field]:
            raise ValueError('Frozen field changed across refreeze: '+field)
    # Verify preflight ZIP (85 exposed, 9067 SFT-only breakdown)
    zpath=Path(args.handoff_zip);zbytes=zpath.read_bytes();zsha=sha(zbytes)
    if zsha!=PR55_HANDOFF_ZIP_SHA: raise ValueError('Preflight ZIP SHA mismatch: '+zsha)
    members={}
    with zipfile.ZipFile(zpath) as zf:
        for name in zf.namelist():
            if not name.endswith('/'): members[name]=zf.read(name)
    preflight_sums={}
    for line in members['SHA256SUMS'].decode('utf-8').strip().splitlines():
        expected,fname=line.split()
        actual=sha(members[fname])
        if actual!=expected: raise ValueError('Preflight SHA256SUMS mismatch: '+fname)
        preflight_sums[fname]=actual
    for name,expected in PR55_DECLARED_HANDOFF_SHAS.items():
        if preflight_sums.get(name)!=expected: raise ValueError('Declared preflight SHA mismatch: '+name)
    elig_audit=load_json(members['eligible-history-audit.json'])
    full_audit=load_json(members['full-history-audit.json'])
    _,elig_all=_audit_history_doc(elig_audit,'eligible-history-audit')
    _,full_all=_audit_history_doc(full_audit,'full-history-audit')
    if set(full_all)!=(set(elig_all)&set(orig_selected)):
        raise ValueError('full-history-audit mismatch with orig_160 restriction')
    id_files={}
    for cls,files in elig_audit['files_by_class'].items():
        for fpath,ids in files.items():
            for tid in ids: id_files.setdefault(tid,[]).append((cls,fpath))
    sft_audit_only=sorted(t for t,es in id_files.items()
                          if all('sft-label-length-audit.json' in p for _,p in es))
    real_exposed=sorted(set(elig_all)-set(sft_audit_only))
    orig_eligible=orig_audit['eligible_task_ids']
    repl_pre=select_replacement_candidates(orig_eligible,orig_selected,elig_all,goals,count=160)
    expected_twins_map={}
    for cand_str,ext_list in repl_pre['cross_goal_instruction_twins_in_remaining'].items():
        for ext_id in ext_list: expected_twins_map[ext_id]=int(cand_str)
    # Verify both new device audit artifacts
    outside_res=verify_outside_candidate_audit(
        args.outside_audit,expected_sha=PR55_OUTSIDE_AUDIT_SHA,expected_targets=(2525,8000))
    unfiltered_res=verify_unfiltered_history_zip(
        args.unfiltered_zip,goals,expected_twins_map,expected_zip_sha=PR55_UNFILTERED_ZIP_SHA)
    if unfiltered_res['goal_map_sha256']!=goals_sha:
        raise ValueError('Unfiltered audit goal_map_sha256 mismatch')
    # Recompute clean 444 pool and top-160 replacement list with confirmed external hits
    all_exposed=set(elig_all)|set(unfiltered_res['actual_hit_ids'])
    repl_final=select_replacement_candidates(
        orig_eligible,orig_selected,elig_all,goals,all_exposed_ids=all_exposed,count=160)
    expected_clean_444=sorted(set(orig_eligible)-set(elig_all)-set(repl_final['excluded_by_instruction_twin']))
    if new_audit['eligible']!=len(expected_clean_444) or new_audit['eligible_task_ids']!=expected_clean_444:
        raise ValueError('Refrozen eligible_task_ids disagrees with recomputed 444 clean pool')
    if new_selected!=repl_final['selected_task_ids'] or head_ids_blob!=canonical_ids_bytes(repl_final['selected_task_ids']):
        raise ValueError('Refrozen holdout_ids.json disagrees with recomputed replacement 160')
    if sha(head_ids_blob)!=PR55_REPLACEMENT_CANDIDATE_SHA or new_audit['candidate_ids_sha256']!=PR55_REPLACEMENT_CANDIDATE_SHA:
        raise ValueError('Refrozen holdout_ids.json SHA mismatch')
    if head_ids_blob!=members['holdout-replacement-candidate.json']:
        raise ValueError('Refrozen holdout_ids.json differs from preflight candidate bytes')
    if new_audit['selected_instruction_sha256']!=repl_final['selected_instruction_sha256']:
        raise ValueError('Refrozen selected_instruction_sha256 mismatch')
    # Verify refreeze metadata block inside holdout-audit.json
    rf=new_audit['refreeze']
    expected_rf=dict(
        original_protocol_commit=REF,
        original_candidate_pool_size=len(orig_eligible),
        original_holdout_sha256=orig_audit['candidate_ids_sha256'],
        device_preflight_evidence_sha256=PR55_HANDOFF_ZIP_SHA,
        unfiltered_history_evidence_sha256=PR55_UNFILTERED_ZIP_SHA,
        device_history_excluded_ids=elig_all,
        sft_audit_only_conservative_exclusion=sft_audit_only[0] if len(sft_audit_only)==1 else sft_audit_only,
        documented_non_sft_usage_excluded_count=len(real_exposed),
        eligible_after_device_exclusions=len(orig_eligible)-len(elig_all),
        outside_candidate_alias_to_candidate={str(k):v for k,v in expected_twins_map.items()},
        outside_candidate_historical_usage_hits=unfiltered_res['actual_hit_ids'],
        additional_same_instruction_exclusions=repl_final['excluded_by_instruction_twin'],
        scanned_device_json_files=unfiltered_res['scanned_json_files'],
        skipped_device_json_files=[],
        kept_original_tasks=repl_final['retained_count'],
        replaced_original_tasks=len(repl_final['added_replacement_ids']),
    )
    for k,v in expected_rf.items():
        if rf.get(k)!=v: raise ValueError('Refreeze metadata mismatch for '+k+': '+str(rf.get(k)))
    # Verify protocol doc invariants across a601999 and 1c19b2b
    doc_refreeze_raw=blob(REF_REFREEZE,'docs/PRM_EXPANDED_FIT_PROTOCOL_20261125.md')
    doc_head_raw=blob(head_ref,'docs/PRM_EXPANDED_FIT_PROTOCOL_20261125.md')
    doc_head_text=doc_head_raw.decode('utf-8')
    doc_checks=dict(
        refreeze_doc_sha256=sha(doc_refreeze_raw),
        head_doc_sha256=sha(doc_head_raw),
        test_authorization_false_present=('`test_authorization=false`' in doc_head_text),
        replacement_sha_present=(PR55_REPLACEMENT_CANDIDATE_SHA in doc_head_text),
        fit501_sha_present=(new_audit['fit501_training_jsonl_sha256'] in doc_head_text),
        historical_3fa1b9d8_link_retained=(REF in doc_head_text),
        team32_1127_mentioned=('11/27' in doc_head_text),
        advances_run_before_presentation=('可立即开展 GPU 采集与训练' in doc_head_text),
        preserves_ppo_and_reserved_test_separate_gate=('PPO 与预留 test 另行验收' in doc_head_text),
    )
    for k in ('test_authorization_false_present','replacement_sha_present','fit501_sha_present',
              'historical_3fa1b9d8_link_retained','team32_1127_mentioned',
              'preserves_ppo_and_reserved_test_separate_gate'):
        if not doc_checks[k]: raise ValueError('Protocol doc invariant failed: '+k)
    # Optional published-artifact cross-check
    pub_section=None
    if args.git_ref or args.archive:
        sources,hits=scan_published_sources(
            args.repo,args.git_ref,args.archive,set(expected_clean_444)|set(expected_twins_map.keys()))
        pub_in_444=sorted(int(k) for k in hits if int(k) in set(expected_clean_444))
        pub_twins=sorted(int(k) for k in hits if int(k) in set(expected_twins_map.keys()))
        if pub_in_444: raise ValueError('Published artifacts hit clean 444 pool: '+str(pub_in_444))
        pub_section=dict(sources=sources,confirmed_in_clean_444=pub_in_444,
                         confirmed_external_twins=pub_twins,hits_by_task_id=hits)
    output=dict(
        reviewed_commits=dict(
            original_protocol=REF,
            timing_gate_1127=REF_TIMING_1127,
            formal_refreeze=REF_REFREEZE,
            pr55_head=head_ref,
        ),
        mode='refreeze-audit',
        scope='Independent pre-execution verification of PR #55 re-freeze (a601999) + schedule update (1c19b2b) and all 3 device history audit artifacts.',
        frozen_artifacts=dict(
            holdout_ids_sha256=sha(head_ids_blob),
            holdout_ids_size_bytes=len(head_ids_blob),
            holdout_audit_sha256=sha(head_audit_blob),
            holdout_audit_size_bytes=len(head_audit_blob),
            dev_rollout_sha256=sha(orig_cfg_blob),
            goal_map_sha256=goals_sha,
            fit501_training_jsonl_sha256=new_audit['fit501_training_jsonl_sha256'],
            fit_order_first_six_ids=new_audit['fit_order_first_six_ids'],
            split_sha256=new_audit['split_sha'],
        ),
        device_artifacts=dict(
            preflight_zip=dict(name=zpath.name,size_bytes=len(zbytes),sha256=zsha,
                               verified_members=len(preflight_sums),
                               exposed_85_count=len(elig_all),
                               sft_audit_only_ids=sft_audit_only,
                               real_non_sft_exposed_count=len(real_exposed)),
            outside_candidate_audit=outside_res,
            unfiltered_history_zip=unfiltered_res,
        ),
        refreeze_recomputation=dict(
            original_eligible_count=len(orig_eligible),
            device_exposed_count=len(elig_all),
            remaining_after_id_exclusion_count=repl_final['remaining_after_id_exclusion_count'],
            excluded_by_instruction_twin=repl_final['excluded_by_instruction_twin'],
            clean_eligible_count=len(expected_clean_444),
            selected_count=repl_final['selected_count'],
            retained_count=repl_final['retained_count'],
            removed_from_original_160=repl_final['removed_from_original_160'],
            added_replacement_ids=repl_final['added_replacement_ids'],
            selected_ids_sha256=repl_final['selected_ids_sha256'],
            matches_refrozen_holdout_ids_bytes=True,
            matches_refrozen_holdout_audit_metadata=True,
            matches_all_160_instruction_hashes=True,
        ),
        protocol_doc_checks=doc_checks,
        published_artifact_crosscheck=pub_section,
        caveats=[
            'B1 SFT inherited exposure is outside PRM holdout isolation and remains disclosed separately.',
            'Task 9067 appears only in copied sft-label-length-audit.json files (not PRM/RL/progress runs) and is conservatively excluded (84 non-SFT + 1 conservative = 85); excluding or keeping 9067 does not affect the top-160 selection.',
            'Intermediate outside-candidate JSON (edfe3ca0...) was a task_id-only scan of [2525, 8000] and is superseded by the broader unfiltered ZIP (0df7049d...) covering all 6 exhaustive goal-map twins across task_id, task_ids, sample_keys, and sample_key.',
            'Commit 1c19b2b advances GPU rollout/training to start immediately after re-freeze and device preflight agreement (before Team 32 presentation on 11/27 afternoon) while preserving test_authorization=false and separate PPO/reserved-test acceptance.',
        ],
        re_freeze_verified=True,
        new_rollout_or_training_performed=False,
        test_authorized=False,
    )
    with Path(args.output).open('x',encoding='utf-8',newline='\n') as handle:
        json.dump(output,handle,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps({
        'mode':output['mode'],'pr55_head':head_ref,
        'holdout_ids_sha256':output['frozen_artifacts']['holdout_ids_sha256'],
        'holdout_audit_sha256':output['frozen_artifacts']['holdout_audit_sha256'],
        'clean_eligible_count':len(expected_clean_444),
        'outside_audit_sha256':outside_res['sha256'],
        'unfiltered_zip_sha256':unfiltered_res['sha256'],
        'unfiltered_hits':unfiltered_res['actual_hit_ids'],
        'excluded_by_twin':repl_final['excluded_by_instruction_twin'],
        're_freeze_verified':True,'test_authorized':False},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('original','replacement-preflight','replacement-handoff','refreeze-audit'),
                        default='original')
    parser.add_argument('--repo',required=True)
    parser.add_argument('--goal-map',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--data-archive')
    parser.add_argument('--probe-archive')
    parser.add_argument('--tokenizer')
    parser.add_argument('--handoff-dir')
    parser.add_argument('--handoff-zip')
    parser.add_argument('--outside-audit')
    parser.add_argument('--unfiltered-zip')
    parser.add_argument('--refreeze-ref',default=REF_HEAD)
    parser.add_argument('--archive',action='append',default=[])
    parser.add_argument('--git-ref',action='append',default=[])
    args=parser.parse_args()
    if Path(args.output).exists(): parser.error('Output must be new')
    if args.mode=='replacement-preflight':
        _run_replacement_preflight(args);return
    if args.mode=='replacement-handoff':
        if not (args.handoff_dir or args.handoff_zip):
            parser.error('--handoff-dir or --handoff-zip is required for replacement-handoff')
        _run_replacement_handoff(args);return
    if args.mode=='refreeze-audit':
        for name in ('handoff_zip','outside_audit','unfiltered_zip'):
            if not getattr(args,name): parser.error('--'+name.replace('_','-')+' is required for refreeze-audit')
        _run_refreeze_audit(args);return
    for name in ('data_archive','probe_archive','tokenizer'):
        if not getattr(args,name): parser.error('--'+name.replace('_','-')+' is required')
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
