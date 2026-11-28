"""Independent saved-output audit for PR #52. No Torch, model or training run."""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import zipfile

SAMPLES_SHA = '9c88beffaef277f519712ab4a07c45766212579770d6292572c7f189bca4d8fe'
TRAIN_SHA = '8c85e56b3ac2e1f0a09b54560ad0f9bf77eedddb241d73617c7ff32416522e20'
PROBE_SHA = 'aa33e5ca0b62e06a43801ea6416326256ae0d389f8ee8e6c3d46cbf571e75b08'
HIDDEN_SHA = 'aa35ec78c4149dc81414e0a44b49740abe5b55103b2fb7fdc57383bc67e5615d'
SOURCE_REF = '5ae6d83553ced1a74db32d750ada6f9d9b6786b1'
CONTINUATION_SHA = {
    48: '4e115935e87cd3b6577e4a5a56a0fc2a1f4fb22b5eefa9a1064fe7fa0b000666',
    80: '4a2daf64b4256d76942b8044d5cade3166c9d4ffa9d40b152cf327931284e46c',
    120: '90b1b1f769f21cfe8b44627ebf628f0c3d2f32b506d19fe7d922cc63585aa49f',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Expected finite numeric evidence')
    return float(value)


def load_json(blob):
    def invalid(value):
        raise ValueError('Non-finite JSON: ' + value)
    return json.loads(blob, parse_constant=invalid)


def verified_archive(path, expected_sha, checksum_manifest=None):
    payload = Path(path).read_bytes()
    if sha(payload) != expected_sha:
        raise ValueError('Archive SHA mismatch')
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(
                PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts
                or '\\' in n for n in names):
            raise ValueError('Unsafe or duplicate archive names')
        manifests = [n for n in names if PurePosixPath(n).name == 'SHA256SUMS']
        if checksum_manifest is not None:
            manifests = [n for n in manifests if n == checksum_manifest]
        if len(manifests) != 1:
            raise ValueError('Expected one archive checksum manifest')
        prefix = manifests[0][:-len('SHA256SUMS')]
        if any(not n.startswith(prefix) for n in names if not n.endswith('/')):
            raise ValueError('Payload outside checksum root')
        files = {n[len(prefix):]: archive.read(n) for n in names if not n.endswith('/')}
    checks = {}
    for line in files['SHA256SUMS'].decode().splitlines():
        digest, name = line.split(None, 1)
        name = name.strip().lstrip('*')
        if name.startswith('./'):
            name = name[2:]
        if name in checks or sha(files[name]) != digest:
            raise ValueError('Payload checksum mismatch: ' + name)
        checks[name] = digest
    if set(checks) != set(files) - {'SHA256SUMS'}:
        raise ValueError('Unlisted archive payload')
    return files, checks


def prediction_map(rows, labels, positions):
    if not isinstance(rows, list) or len(rows) != len(labels):
        raise ValueError('Prediction set is incomplete')
    result = {}
    for row in rows:
        task = row['task_id']
        if type(task) is not int or task not in labels or task in result:
            raise ValueError('Unexpected or duplicated task')
        if row['turn_end_indices'] != positions[task]:
            raise ValueError('Wrong scoring positions')
        scores = row['step_scores']
        if len(scores) != len(positions[task]):
            raise ValueError('Incomplete step scores')
        values = [finite(v) for v in scores]
        total = finite(row['total'])
        if not math.isclose(sum(values), total, abs_tol=1e-5, rel_tol=1e-6):
            raise ValueError('Step scores do not sum to recorded total')
        result[task] = {'total': total, 'steps': values}
    return result


def score(values, labels):
    errors = [finite(values[t]) - finite(labels[t]) for t in labels]
    return {'mse': sum(e*e for e in errors)/len(errors),
            'max_absolute_error': max(abs(e) for e in errors)}


def validate_continuation(summary, previous, expected_final):
    """Check saved linkage, not a fresh deserialization of optimizer/RNG state."""
    if (summary['identity'] != previous['identity'] or
            summary['optimization'] != previous['optimization'] or
            summary['parameters_before'] != previous['parameters_after']):
        raise ValueError('Continuation identity/settings/parameter linkage mismatch')
    expected = list(range(previous['optimizer_steps'] + 1, expected_final + 1))
    if (summary['optimizer_steps'] != expected_final or
            [r['optimizer_step'] for r in summary['records']] != expected):
        raise ValueError('Continuation segment has a gap or wrong budget')


def group_error_components(values, labels, positions):
    """Exact MSE split by turn count: group mean error + centered error."""
    mean_component = centered_component = 0.0
    for turns in set(map(len, positions.values())):
        tasks = [t for t in labels if len(positions[t]) == turns]
        pred_mean = sum(values[t] for t in tasks) / len(tasks)
        label_mean = sum(labels[t] for t in tasks) / len(tasks)
        mean_component += len(tasks) * (pred_mean-label_mean)**2 / len(labels)
        centered_component += sum(((values[t]-pred_mean)-(labels[t]-label_mean))**2
                                  for t in tasks) / len(labels)
    return {'group_mean_error_mse': mean_component,
            'within_group_centered_error_mse': centered_component}


def audit_run(summary, reference, reload, labels, positions):
    ref = prediction_map(reference, labels, positions)
    restored = prediction_map(reload['predictions'], labels, positions)
    if (reload.get('reference_match') is not True or reload.get('rejected') != [] or
            reload.get('total_samples') != len(labels) or
            reload['identity'] != summary['identity']):
        raise ValueError('Reload metadata or identity mismatch')
    step_difference = total_difference = 0.0
    for task in labels:
        step_difference = max(step_difference, max(abs(a-b) for a,b in
            zip(ref[task]['steps'], restored[task]['steps'])))
        total_difference = max(total_difference, abs(ref[task]['total']-restored[task]['total']))
        for a,b in zip(ref[task]['steps']+[ref[task]['total']],
                       restored[task]['steps']+[restored[task]['total']]):
            if not math.isclose(a,b,abs_tol=1e-5,rel_tol=1e-5):
                raise ValueError('Reload prediction mismatch')
    if (summary['samples_sha256'] != SAMPLES_SHA or summary['train_ids_sha256'] != TRAIN_SHA or
            summary['total_samples'] != len(labels) or summary['retained_samples'] != len(labels) or
            summary['rejected'] != []):
        raise ValueError('Sample identity/coverage mismatch')
    options = summary['optimization']
    if (options['gradient_accumulation_steps'] != 8 or options['weight_decay'] != .01 or
            options['max_grad_norm'] != 1.0):
        raise ValueError('Unapproved accumulation/optimizer settings')
    records = summary['records']
    if not records:
        raise ValueError('No update records')
    first, final = records[0]['optimizer_step'], summary['optimizer_steps']
    if [r['optimizer_step'] for r in records] != list(range(first, final+1)):
        raise ValueError('Missing/repeated optimizer steps')
    clip_count = 0
    loss_difference = 0.0
    for row in records:
        if row['task_ids'] != list(labels) or row['terminal_rewards'] != list(labels.values()):
            raise ValueError('Training task/label order mismatch')
        if row['samples_consumed'] != 8*row['optimizer_step']:
            raise ValueError('Wrong sample-presentation counter')
        norm, loss = finite(row['gradient_norm']), finite(row['loss'])
        if type(row['gradient_clipped']) is not bool or row['gradient_clipped'] != (norm > 1):
            raise ValueError('Incorrect clipping flag')
        clip_count += row['gradient_clipped']
        before = row['totals_before_update']
        if len(before) != len(labels):
            raise ValueError('Incomplete pre-update predictions')
        expected_loss = score(dict(zip(labels, before)), labels)['mse']
        if not math.isclose(loss, expected_loss, abs_tol=1e-5, rel_tol=1e-6):
            raise ValueError('Training loss differs from predictions and labels')
        loss_difference = max(loss_difference, abs(loss-expected_loss))
    if summary['samples_consumed'] != 8*final:
        raise ValueError('Final exposure count mismatch')
    values = {task: ref[task]['total'] for task in labels}
    metrics = score(values, labels)
    full = (values[0]+values[6225])/2
    zero = (values[95]+values[5031])/2
    return dict(metrics, memory_gate_passed=metrics['mse'] < .001 and metrics['max_absolute_error'] < .1,
                cumulative_optimizer_steps=final, segment_optimizer_steps=len(records),
                cumulative_presentations=summary['samples_consumed'],
                segment_clip_count=clip_count, segment_clip_ratio=clip_count/len(records),
                covered_tasks=len(ref), covered_scoring_positions=sum(map(len,positions.values())),
                saved_reload_max_step_difference=step_difference,
                saved_reload_max_total_difference=total_difference,
                training_loss_max_replay_difference=loss_difference,
                focus=[{'task_id': t,'label':labels[t],'total':values[t]} for t in (0,6225,95,5031)],
                full_reward_group_mean=full, zero_reward_group_mean=zero,
                full_minus_zero_group_mean=full-zero,
                recorded_source_commit=summary['identity']['source_commit'],
                peak_allocated_gib=summary['peak_allocated_gib'],
                peak_reserved_gib=summary['peak_reserved_gib'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('probe-archive','features-archive','tokenizer','repo','output'):
        parser.add_argument('--'+name, required=True)
    for step in CONTINUATION_SHA:
        parser.add_argument('--step'+str(step)+'-archive')
    args = parser.parse_args(argv)
    if args.step80_archive and not args.step48_archive:
        parser.error('Step80 audit requires the Step48 predecessor archive')
    if args.step120_archive and not args.step80_archive:
        parser.error('Step120 audit requires the Step80 predecessor archive')
    if Path(args.output).exists():
        parser.error('Output must be new')
    probe, probe_hashes = verified_archive(args.probe_archive, PROBE_SHA)
    features, feature_hashes = verified_archive(args.features_archive, HIDDEN_SHA)
    raw = probe['lrprobe/samples.jsonl']
    if sha(raw) != SAMPLES_SHA:
        raise ValueError('Wrong eight original inputs')
    samples = [load_json(s) for s in raw.splitlines() if s]
    labels = {r['task_id']:r['terminal_reward'] for r in samples}
    if len(labels) != 8:
        raise ValueError('Wrong original sample count')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    identity = load_json(probe['lrprobe/hidden-lr-1e5-probe/summary.json'])['identity']
    for name,digest in identity['tokenizer_sha256'].items():
        if sha((Path(args.tokenizer)/name).read_bytes()) != digest:
            raise ValueError('Tokenizer file SHA mismatch')
    if sha(tok.chat_template.encode()) != identity['template_sha256']:
        raise ValueError('Tokenizer template mismatch')
    positions = {}
    for sample in samples:
        messages = sample['messages']
        ids = list(tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
        if len(ids)>2048:
            raise ValueError('Overlong original input')
        ends=[]
        for i,m in enumerate(messages):
            if m['role']!='assistant': continue
            prefix=list(tok.apply_chat_template(messages[:i+1],tokenize=True,add_generation_prompt=False))
            if ids[:len(prefix)] != prefix: raise ValueError('Non-prefix-stable template')
            ends.append(max(k for k,v in enumerate(prefix) if v==151645))
        positions[sample['task_id']]=ends
    source_checks={}
    for name in ('train.py','infer.py','runtime.py','model_core.py','train_core.py','config.py','tokenize.py'):
        pinned=subprocess.check_output(['git','show',SOURCE_REF+':src/webshop_lab/prm/'+name],cwd=args.repo)
        if sha(pinned)!=sha(features['source/'+name]):
            raise ValueError('Published source differs from execution-order source: '+name)
        source_checks[name]=sha(pinned)
    runs={}
    initial_parameters=None
    checkpoint_hashes=0
    for arm in ('hidden-lr-1e7-probe','hidden-lr-1e6-probe','hidden-lr-1e5-probe'):
        summary=load_json(probe['lrprobe/'+arm+'/summary.json'])
        expected_lr={'hidden-lr-1e7-probe':1e-7,'hidden-lr-1e6-probe':1e-6,
                     'hidden-lr-1e5-probe':1e-5}[arm]
        if summary['optimization']['learning_rate']!=expected_lr:
            raise ValueError('Arm name and actual learning rate differ')
        if initial_parameters is None:
            initial_parameters=summary['parameters_before']
        if summary['parameters_before']!=initial_parameters:
            raise ValueError('Arms did not start from matching parameter digests')
        if summary['identity']!=identity or summary['optimizer_steps']!=16:
            raise ValueError('Unexpected short-probe identity or update budget')
        if (summary['identity']['policy_dtype']!='float32' or
                summary['identity']['max_length']!=2048 or
                summary['identity']['initialization']['sft_adapter_files_sha256'][
                    'adapter_model.safetensors']!=
                'c39e2a78e7c5973328e9d2c23fb6320e566352e63cc089562488f3dd46ae5f9c'):
            raise ValueError('Wrong FP32/context/B1 identity')
        if any(source_checks[n]!=d for n,d in summary['identity']['source_sha256'].items()):
            raise ValueError('Recorded core source hash differs from supplied source')
        reference=load_json(probe['lrprobe/'+arm+'/reference-predictions.json'])
        reload=load_json(probe['lrprobe/'+arm+'-reload.json'])
        runs[arm]=audit_run(summary,reference,reload,labels,positions)
        manifest=load_json(probe['lrprobe/'+arm+'/prm-manifest.json'])
        if manifest['identity']!=summary['identity'] or manifest['schema_version']!=2:
            raise ValueError('Checkpoint identity/schema differs from summary')
        if (manifest['architecture']['representation']!='final_hidden_state' or
                manifest['architecture']['input_dim']!=1536 or
                manifest['score_definition']!='assistant_im_end_final_norm_hidden_linear_sum_mse'):
            raise ValueError('Unexpected scorer architecture/definition')
    manifest=load_json(probe['lrprobe/hidden-lr-1e5-probe-step16/prm-manifest.json'])
    for name,digest in manifest['files_sha256'].items():
        if sha(probe['lrprobe/hidden-lr-1e5-probe-step16/'+name])!=digest:
            raise ValueError('Step16 checkpoint payload mismatch: '+name)
        checkpoint_hashes+=1
    mean=sum(labels.values())/8
    constant={t:mean for t in labels}
    grouped={n:sum(labels[t] for t in labels if len(positions[t])==n)/sum(len(positions[t])==n for t in labels)
             for n in set(map(len,positions.values()))}
    baseline={'scope':'in-sample references for these eight; not independent dev baselines',
              'constant':score(constant,labels),
              'turn_count_group_mean':score({t:grouped[len(positions[t])] for t in labels},labels)}
    previous = load_json(probe['lrprobe/hidden-lr-1e5-probe/summary.json'])
    previous_ref = load_json(probe['lrprobe/hidden-lr-1e5-probe/reference-predictions.json'])
    previous_values = {r['task_id']: r['total'] for r in previous_ref}
    continuations = {}
    for step, archive_sha in CONTINUATION_SHA.items():
        archive_path = getattr(args, 'step'+str(step)+'_archive')
        if not archive_path:
            continue
        files, hashes = verified_archive(archive_path, archive_sha)
        if sha(files['samples.jsonl']) != SAMPLES_SHA or sha(files['train_ids.json']) != TRAIN_SHA:
            raise ValueError('Continuation input file bytes differ')
        summary = load_json(files['checkpoint/summary.json'])
        validate_continuation(summary, previous, step)
        reference = load_json(files['checkpoint/reference-predictions.json'])
        reload = load_json(files['hidden-1e5-acc8-step'+str(step)+'-reload.json'])
        result = audit_run(summary, reference, reload, labels, positions)
        manifest = load_json(files['checkpoint/prm-manifest.json'])
        old_manifest = load_json(probe['lrprobe/hidden-lr-1e5-probe-step16/prm-manifest.json'])
        for field in ('identity', 'schema_version', 'architecture', 'score_definition'):
            if manifest[field] != old_manifest[field]:
                raise ValueError('Continuation checkpoint contract changed')
        if set(manifest['files_sha256']) != set(old_manifest['files_sha256']):
            raise ValueError('Incomplete checkpoint payload manifest')
        for name, digest in manifest['files_sha256'].items():
            if sha(files['checkpoint/'+name]) != digest:
                raise ValueError('Continuation checkpoint payload SHA mismatch: '+name)
        values = {r['task_id']: r['total'] for r in reference}
        before_components = group_error_components(previous_values, labels, positions)
        after_components = group_error_components(values, labels, positions)
        decline = score(previous_values, labels)['mse'] - result['mse']
        result.update(archive_sha256=archive_sha, verified_archive_payloads=len(hashes),
                      verified_checkpoint_payloads=len(manifest['files_sha256']),
                      predecessor_optimizer_steps=previous['optimizer_steps'],
                      saved_parameter_digest_linkage_matches=True,
                      optimizer_and_rng_state_deserialized_here=False,
                      mse_components=after_components,
                      mse_decline_from_predecessor=decline,
                      group_mean_share_of_mse_decline=(before_components['group_mean_error_mse']-
                          after_components['group_mean_error_mse'])/decline if decline else None,
                      per_sample=[dict(task_id=t, label=labels[t], total=values[t],
                          error=values[t]-labels[t], previous_total=previous_values[t]) for t in labels])
        continuations[str(step)] = result
        previous, previous_values = summary, values
    report={'scope':'Independent CPU arithmetic of saved GPU outputs; no model reload or new training',
            'probe_archive_sha256':PROBE_SHA,'features_archive_sha256':HIDDEN_SHA,
            'verified_probe_payloads':len(probe_hashes),'verified_feature_payloads':len(feature_hashes),
            'verified_step16_checkpoint_payloads':checkpoint_hashes,
            'ordered_sample_ids':list(labels),'samples_sha256':SAMPLES_SHA,
            'real_tokenizer_scoring_positions_recomputed':True,'source_file_sha256':source_checks,
            'source_limit':'Seven supplied core files match pinned PR52; full executed Git tree not independently recovered',
            'baselines':baseline,'runs':runs,
            'independent_model_reload_performed_here':False,
            'continuations':continuations,
            'pending':'Only supplied archived checkpoints audited; subsequent checkpoints require separate acceptance',
            'test_authorized':False}
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x',encoding='utf-8',newline='\n') as out:
        json.dump(report,out,ensure_ascii=False,indent=2,allow_nan=False);out.write('\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
