"""Audit one real on-policy PPO update using the verified project implementation."""
import argparse
import datetime
import json
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('stage', 'config', 'ids', 'split-dir', 'goal-map', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    import torch
    from webshop_lab import ppo
    from webshop_lab.common import file_hash, git_head
    from webshop_lab.model import precision_record

    stage, out = args.stage, args.output
    audit_path = out.with_name(out.name + '.audit.json')
    if audit_path.exists() or out.exists():
        raise ValueError('Refusing to overwrite a previous training attempt')
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    experiment = json.loads((stage / 'experiment-plan.json').read_text(encoding='utf-8'))
    publication = json.loads((stage / 'publication.json').read_text(encoding='utf-8'))
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    ids = json.loads(args.ids.read_text(encoding='utf-8'))
    assert cfg['policy_dtype'] == 'float32' and not cfg['load_in_4bit']
    assert git_head() == experiment['execution_source_commit']
    assert not subprocess.check_output(
        ['git', 'status', '--porcelain', '--untracked-files=no'],
        cwd=experiment['execution_source_dir'], text=True).strip()
    assert file_hash(__file__) == publication['driver_sha256']['run-update.py']
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert not torch.is_autocast_enabled()
    health = ppo.request(cfg['environment_url'].rstrip('/') + '/health')
    assert health.get('type_reward_cache') == 'page-local-v1'
    assert health.get('option_state') is True
    torch.cuda.reset_peak_memory_stats()
    adapter = args.resume / 'adapter' if args.resume else Path(cfg['adapter'])
    input_paths = {
        'base_weights': Path(cfg['model']) / 'model.safetensors',
        'initial_sft_weights': Path(cfg['adapter']) / 'adapter_model.safetensors',
        'initial_sft_config': Path(cfg['adapter']) / 'adapter_config.json',
        'loaded_adapter_weights': adapter / 'adapter_model.safetensors',
        'loaded_adapter_config': adapter / 'adapter_config.json',
        'config': args.config, 'ids': args.ids, 'goal_map': args.goal_map,
        'experiment_plan': stage / 'experiment-plan.json',
    }
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'generation_config.json'):
        path = Path(cfg['model']) / name
        if path.is_file():
            input_paths['model_aux_' + name] = path
    if args.resume:
        input_paths['loaded_critic'] = args.resume / 'critic.pt'
        input_paths['loaded_training_state'] = args.resume / 'training-state.pt'
    hashes = {name: file_hash(path) for name, path in input_paths.items()}
    assert hashes['base_weights'] == cfg['base_weights_sha256']
    assert hashes['initial_sft_weights'] == cfg['adapter_sha256']
    audit = {
        'scope': 'new full-coverage on-policy training update',
        'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_commit': git_head(), 'specification_commit': publication['head'],
        'driver_sha256': file_hash(__file__), 'input_sha256': hashes,
        'task_ids': ids, 'resume': str(args.resume) if args.resume else None,
        'backend': {
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'autocast_enabled': torch.is_autocast_enabled(),
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'device': torch.cuda.get_device_name(0),
        },
        'status': 'running', 'test_authorization': False,
    }

    def save():
        audit_path.write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')

    save()
    original_load, original_update, original_write = ppo.load, ppo.update, ppo.write_json

    def audited_load(*positional, **keywords):
        model, tokenizer, dtype = original_load(*positional, **keywords)
        assert dtype == torch.float32
        audit['base_precision'] = precision_record(model, dtype)
        assert set(audit['base_precision']['parameter_numel_by_dtype']) == {'torch.float32'}
        save()
        return model, tokenizer, dtype

    def audited_update(model, critic, turns, optimizer, parameters):
        audit['policy_precision_at_update'] = precision_record(model, torch.float32)
        audit['critic_dtypes'] = sorted({str(p.dtype) for p in critic.parameters()})
        assert set(audit['policy_precision_at_update']['parameter_numel_by_dtype']) == {'torch.float32'}
        assert audit['critic_dtypes'] == ['torch.float32']
        audit['collected_turns'] = len(turns)
        save()
        return original_update(model, critic, turns, optimizer, parameters)

    def audited_write(path, value):
        if Path(path).name == 'run.meta.json':
            value = {**value, 'experiment_scope': 'full-coverage-rl-20261123-v1',
                     'execution_driver_sha256': audit['driver_sha256'],
                     'specification_commit': publication['head'],
                     'input_sha256_at_load': hashes, 'backend': audit['backend'],
                     'loaded_base_precision': audit.get('base_precision')}
        original_write(path, value)

    ppo.load, ppo.update, ppo.write_json = audited_load, audited_update, audited_write
    start = time.monotonic()
    try:
        ppo.run(args)
        after = {name: file_hash(path) for name, path in input_paths.items()}
        assert after == hashes, 'Training input changed during the update'
        summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
        audit.update(status='complete', run_id=summary['run_id'], summary=summary,
                     input_unchanged=True, summary_sha256=file_hash(out / 'summary.json'))
    except BaseException as error:
        audit.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        audit.update(elapsed_seconds=time.monotonic() - start,
                     ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024 ** 3,
                     peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024 ** 3,
                     device_total_gib=torch.cuda.get_device_properties(0).total_memory / 1024 ** 3)
        save()
        print(json.dumps({'status': audit['status'], 'output': str(out),
                          'elapsed_seconds': audit['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
