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
from .train_core import optimizer_step


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('config','samples','train-ids'):
        parser.add_argument('--'+key,required=True,type=Path)
    parser.add_argument('--steps',type=int,default=2)
    parser.add_argument('--learning-rate',type=float,default=1e-5)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    args = parser.parse_args()
    if args.steps < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('更新步数和学习率必须为正')
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
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=args.learning_rate,weight_decay=0.)
        completed = 0
        data_sha = sha(args.samples)
        if args.resume:
            state = torch.load(args.resume/'training-state.pt',map_location='cpu',weights_only=True)
            if (state['samples_sha256'] != data_sha or state['train_ids_sha256'] != train_ids_sha
                    or state['seed'] != args.seed or state['learning_rate'] != args.learning_rate):
                raise ValueError('续训数据、种子或学习率改变')
            optimizer.load_state_dict(state['optimizer'])
            completed = state['completed_steps']
            torch.set_rng_state(state['cpu_rng'])
            if args.device == 'cuda':
                torch.cuda.set_rng_state_all(state['cuda_rng'])
        records = []
        before_parameters={kind:parameter_digest(kind) for kind in ('head','lora')}
        for offset in range(args.steps):
            index = (completed+offset) % len(rows)
            task_id,row = rows[index]
            batch = tensor_batch([row],tok.pad_token_id,args.device,True)
            report = optimizer_step(model,batch,optimizer)
            report['task_id'] = task_id
            report['optimizer_step'] = completed+offset+1
            report['head_gradient_l1'] = float(model.score_head.weight.grad.detach().abs().sum())
            report['lora_gradient_l1'] = sum(float(p.grad.detach().abs().sum()) for n,p in model.named_parameters()
                                           if 'lora_' in n and p.grad is not None)
            report['terminal_reward'] = row['terminal_reward']
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
            'train_ids_sha256':train_ids_sha,'seed':args.seed,'learning_rate':args.learning_rate},
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
            'optimizer_steps':completed+args.steps,'records':records,
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
