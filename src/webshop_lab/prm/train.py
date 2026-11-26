"""真实PRM短训入口；输入为数据负责人转换后的完整train轨迹。"""
import argparse
from contextlib import contextmanager
import datetime
import json
import math
import random
import re
import time
from pathlib import Path
import torch
from .config import load_config
from .tokenize import tokenize_trajectory, TrajectoryTooLong
from .runtime import build_prm, tensor_batch, sha, write_json, finalize_manifest, verify_runtime_inputs
from .model_core import save_prm
from .train_core import accumulated_optimizer_step


@contextmanager
def attempt_record(output,device):
    path=output.with_name(output.name+'.attempt.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    record={'status':'running','device':device,
            'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'test_authorization':False}
    with path.open('x',encoding='utf-8') as stream:
        json.dump(record,stream,ensure_ascii=False,indent=2)
    try:
        yield
    except BaseException as error:
        record.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise
    else:
        record['status']='complete'
    finally:
        record['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        if device == 'cuda' and torch.cuda.is_available():
            record['peak_allocated_gib']=torch.cuda.max_memory_allocated()/1024**3
            record['peak_reserved_gib']=torch.cuda.max_memory_reserved()/1024**3
        write_json(path,record)


def load_samples(path, tokenizer, max_length, allowed_ids=None, training=False, expected_policy_sha=None):
    rows = []
    rejected = []
    raw_rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    for index,row in enumerate(raw_rows):
        if type(row.get('task_id')) is not int or not isinstance(row.get('messages'),list):
            raise ValueError('每条样本需显式task_id和完整messages；不是原始rollout直读入口')
        if allowed_ids is not None and row['task_id'] not in allowed_ids:
            raise ValueError('PRM拟合样本不在冻结策略train池')
        if training:
            if not re.fullmatch('[0-9a-f]{64}',str(row.get('source_sha256',''))):
                raise ValueError('拟合轨迹需保留原件SHA')
            if expected_policy_sha is not None and row.get('policy_adapter_sha256') != expected_policy_sha:
                raise ValueError('探索轨迹行为策略身份不是配置声明的原B1')
            reward = row.get('terminal_reward')
            if type(reward) not in (int,float) or not math.isfinite(reward) or not 0 <= reward <= 1:
                raise ValueError('缺少有效终局任务标签')
            if type(row.get('done')) is not bool or row.get('termination') not in ('purchase','max_steps'):
                raise ValueError('数据层须明确购买或正常步数结束；技术中断不能当失败拟合')
            if row['done'] != (row['termination'] == 'purchase') or (not row['done'] and reward != 0):
                raise ValueError('购买状态、正常结束类型与终局任务奖励矛盾')
        try:
            encoded = tokenize_trajectory(tokenizer,row['messages'],max_length)
        except TrajectoryTooLong as error:
            rejected.append({'row':index,'task_id':row['task_id'],'reason':str(error),
                             'terminal_reward':row.get('terminal_reward')})
            continue
        if training:
            encoded['terminal_reward'] = row['terminal_reward']
        rows.append((row['task_id'],encoded))
    return rows,rejected,len(raw_rows)


def predict(model,tokenizer,rows):
    model.eval()
    result = []
    with torch.no_grad():
        for task_id,row in rows:
            batch = tensor_batch([row],tokenizer.pad_token_id,next(model.parameters()).device,False)
            out = model(**batch)
            result.append({'task_id':task_id,'turn_end_indices':row['turn_end_indices'],
                           'step_scores':out['step_scores'][0].cpu().tolist(),
                           'total':float(out['totals'][0])})
    return result


def cosine_lr_multiplier(step_index, total_steps):
    if type(step_index) is not int or type(total_steps) is not int or step_index < 0 or total_steps < 1:
        raise ValueError('余弦退火步数索引与总步数无效')
    if step_index >= total_steps:
        raise ValueError('当前优化步数已超出余弦退火总步数上限')
    return 0.5 * (1.0 + math.cos(math.pi * step_index / total_steps))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('config','samples','train-ids'):
        parser.add_argument('--'+key,required=True,type=Path)
    parser.add_argument('--steps', type=int, default=2, help='Number of optimizer updates')
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                        help='Base learning rate for LoRA (and score_head when --head-learning-rate is omitted)')
    parser.add_argument('--head-learning-rate', type=float, default=None,
                        help='Optional separate base learning rate for score_head parameters')
    parser.add_argument('--lr-scheduler', choices=('constant', 'cosine'), default='constant')
    parser.add_argument('--cosine-total-steps', type=int, default=None,
                        help='Total optimizer updates for cosine decay; defaults to --steps on fresh runs')
    parser.add_argument('--weight-decay', type=float, default=0.)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='Complete one-trajectory microbatches per optimizer update')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    args = parser.parse_args()
    if args.steps < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('更新步数和学习率必须为正')
    if args.head_learning_rate is not None and (not math.isfinite(args.head_learning_rate) or args.head_learning_rate <= 0):
        raise ValueError('打分头学习率必须为有限正数')
    if args.gradient_accumulation_steps < 1 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError('梯度累积须大于零，权重衰减须有限且非负')
    cosine_total_steps = None
    if args.lr_scheduler == 'cosine':
        cosine_total_steps = args.steps if args.cosine_total_steps is None else args.cosine_total_steps
        if type(cosine_total_steps) is not int or cosine_total_steps < 1:
            raise ValueError('余弦退火总步数必须为正整数')
    elif args.cosine_total_steps is not None:
        raise ValueError('constant 调度不允许指定 cosine_total_steps')
    cfg = load_config(args.config)
    out = Path(cfg.output_dir)
    if out.exists():
        raise ValueError('输出目录已存在，禁止覆盖')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    with attempt_record(out,args.device):
        model,tok,identity,inputs = build_prm(cfg,args.resume,args.device,True)
        def parameter_digest(kind):
            import hashlib
            digest=hashlib.sha256()
            for name,p in sorted(model.named_parameters()):
                if (kind == 'head' and name.startswith('score_head.')) or (kind == 'lora' and 'lora_' in name):
                    digest.update(name.encode())
                    digest.update(p.detach().float().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()
        allowed = set(json.loads(args.train_ids.read_text(encoding='utf-8')))
        if not allowed or any(type(i) is not int for i in allowed):
            raise ValueError('训练任务清单无效')
        train_ids_sha = sha(args.train_ids)
        if args.device == 'cuda' and train_ids_sha != '8c85e56b3ac2e1f0a09b54560ad0f9bf77eedddb241d73617c7ff32416522e20':
            raise ValueError('GPU实训必须使用当前冻结策略train清单')
        rows,rejected,total = load_samples(args.samples,tok,cfg.max_length,allowed,True,
            cfg.sft_adapter_files_sha256.get('adapter_model.safetensors'))
        if not rows:
            raise ValueError('无完整可训练轨迹；检查超长拒绝和数据来源')
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if args.head_learning_rate is None:
            base_group_lrs = [args.learning_rate]
            param_groups = [{'params': [p for _, p in trainable], 'lr': args.learning_rate}]
        else:
            head_params = [p for n, p in trainable if n.startswith('score_head.')]
            lora_params = [p for n, p in trainable if 'lora_' in n]
            if not head_params or not lora_params or len(head_params) + len(lora_params) != len(trainable):
                raise ValueError('独立打分头学习率要求明确分离非空 score_head 与 LoRA 参数组')
            base_group_lrs = [args.head_learning_rate, args.learning_rate]
            param_groups = [
                {'params': head_params, 'lr': args.head_learning_rate},
                {'params': lora_params, 'lr': args.learning_rate},
            ]
        optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate, weight_decay=args.weight_decay)
        accumulation = args.gradient_accumulation_steps
        completed = 0
        consumed = 0
        data_sha = sha(args.samples)
        if args.resume:
            state = torch.load(args.resume/'training-state.pt',map_location='cpu',weights_only=True)
            if (state['samples_sha256'] != data_sha or state['train_ids_sha256'] != train_ids_sha
                    or state['seed'] != args.seed or state['learning_rate'] != args.learning_rate):
                raise ValueError('续训数据、种子或学习率改变')
            completed = state['completed_steps']
            prior_accumulation = state.get('gradient_accumulation_steps', 1)
            consumed = state.get('samples_consumed', completed)
            prior_head_lr = state.get('head_learning_rate')
            prior_scheduler = state.get('lr_scheduler', 'constant')
            prior_cosine_steps = state.get('cosine_total_steps')
            groups = state['optimizer']['param_groups']
            expected_saved_lrs = (
                [lr * cosine_lr_multiplier(completed - 1, prior_cosine_steps) for lr in base_group_lrs]
                if (prior_scheduler == 'cosine' and type(completed) is int and completed >= 1
                    and type(prior_cosine_steps) is int and completed <= prior_cosine_steps)
                else base_group_lrs
            )
            if (type(completed) is not int or completed < 0 or type(consumed) is not int
                    or type(prior_accumulation) is not int or prior_accumulation != accumulation
                    or consumed != completed * accumulation
                    or prior_head_lr != args.head_learning_rate
                    or prior_scheduler != args.lr_scheduler
                    or prior_cosine_steps != cosine_total_steps
                    or len(groups) != len(base_group_lrs)
                    or any(g['weight_decay'] != args.weight_decay for g in groups)
                    or any(not math.isclose(g['lr'], exp_lr, rel_tol=1e-12, abs_tol=0.0)
                           for g, exp_lr in zip(groups, expected_saved_lrs))):
                raise ValueError('续训梯度累积、样本计数或 AdamW 超参数不一致')
            optimizer.load_state_dict(state['optimizer'])
            torch.set_rng_state(state['cpu_rng'])
            if args.device == 'cuda':
                torch.cuda.set_rng_state_all(state['cuda_rng'])
        if args.lr_scheduler == 'cosine' and completed + args.steps > cosine_total_steps:
            raise ValueError('优化总步数超出 cosine_total_steps 上限')
        effective_base_head_lr = args.head_learning_rate if args.head_learning_rate is not None else args.learning_rate
        records = []
        before_parameters={kind:parameter_digest(kind) for kind in ('head','lora')}
        for offset in range(args.steps):
            step_index = completed + offset
            lr_mult = cosine_lr_multiplier(step_index, cosine_total_steps) if args.lr_scheduler == 'cosine' else 1.0
            for group, base_lr in zip(optimizer.param_groups, base_group_lrs):
                group['lr'] = base_lr * lr_mult
            selected = [rows[(consumed + i) % len(rows)] for i in range(accumulation)]
            batches = [tensor_batch([row], tok.pad_token_id, args.device, True) for _, row in selected]
            report = accumulated_optimizer_step(model, batches, optimizer)
            consumed += len(selected)
            report['task_ids'] = [task_id for task_id, _ in selected]
            report['terminal_rewards'] = [row['terminal_reward'] for _, row in selected]
            report['samples_consumed'] = consumed
            report['gradient_clipped'] = report['gradient_norm'] > 1.
            report['learning_rate'] = args.learning_rate * lr_mult
            report['head_learning_rate'] = effective_base_head_lr * lr_mult
            if accumulation == 1:
                report['task_id'] = report['task_ids'][0]
                report['terminal_reward'] = report['terminal_rewards'][0]
            report['optimizer_step'] = step_index + 1
            report['head_gradient_l1'] = float(model.score_head.weight.grad.detach().abs().sum())
            report['lora_gradient_l1'] = sum(float(p.grad.detach().abs().sum()) for n,p in model.named_parameters()
                                           if 'lora_' in n and p.grad is not None)
            records.append(report)
        if not all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) for r in records):
            raise ValueError('训练含非有限记录')
        if verify_runtime_inputs(cfg) != inputs or sha(args.samples) != data_sha or sha(args.train_ids) != train_ids_sha:
            raise ValueError('训练过程中输入文件改变')
        reference = predict(model,tok,rows)
        after_parameters={kind:parameter_digest(kind) for kind in ('head','lora')}
        save_prm(model,out,identity)
        torch.save({'optimizer':optimizer.state_dict(),'cpu_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if args.device == 'cuda' else [],
            'completed_steps':completed+args.steps,'samples_sha256':data_sha,
            'train_ids_sha256':train_ids_sha,'seed':args.seed,'learning_rate':args.learning_rate,
            'head_learning_rate':args.head_learning_rate,'lr_scheduler':args.lr_scheduler,
            'cosine_total_steps':cosine_total_steps,
            'gradient_accumulation_steps':accumulation,'samples_consumed':consumed},
            out/'training-state.pt')
        write_json(out/'config.json',cfg.to_dict())
        write_json(out/'reference-predictions.json',reference)
        summary = {'scope':'神经PRM短训；不自动通过正式泛化验收','device':args.device,
            'torch_version':torch.__version__,'cuda_version':torch.version.cuda,
            'input_files_sha256':inputs,'identity':identity,
            'backend':{'float32_matmul_precision':torch.get_float32_matmul_precision(),
                       'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
                       'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
                       'autocast_enabled':torch.is_autocast_enabled()},
            'optimizer_steps':completed+args.steps,'samples_consumed':consumed,
            'optimization':{'microbatch_size':1,'gradient_accumulation_steps':accumulation,
                            'learning_rate':args.learning_rate,
                            'head_learning_rate':effective_base_head_lr,
                            'separate_head_learning_rate':args.head_learning_rate is not None,
                            'lr_scheduler':args.lr_scheduler,
                            'cosine_total_steps':cosine_total_steps,
                            'weight_decay':args.weight_decay,
                            'max_grad_norm':1.0},
            'records':records,
            'samples_sha256':data_sha,'train_ids_sha256':train_ids_sha,
            'total_samples':total,'retained_samples':len(rows),'rejected':rejected,
            'parameters_before':before_parameters,'parameters_after':after_parameters,
            'head_changed':before_parameters['head'] != after_parameters['head'],
            'lora_changed':before_parameters['lora'] != after_parameters['lora'],
            'parameter_numel_by_dtype':{'torch.float32':sum(p.numel() for p in model.parameters())},
            'elapsed_seconds':time.monotonic()-start,'test_authorization':False,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3 if args.device == 'cuda' else None,
            'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3 if args.device == 'cuda' else None}
        write_json(out/'summary.json',summary)
        finalize_manifest(out)
        print(json.dumps(summary,ensure_ascii=False))


if __name__ == '__main__':
    main()
