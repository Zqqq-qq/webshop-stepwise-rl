"""共用runtime完整重载；支持转换后samples及原始rollout，推理均不读取标签。"""
import argparse
import json
from pathlib import Path
from .config import load_config
from .runtime import build_prm, write_json, verify_runtime_inputs
from .train import load_samples, predict
from .predictions import compare_predictions, validate_predictions


def load_rollout_samples(path,tokenizer,max_length,task_id=None,repeat=None):
    # 保留How的原始rollout适配，禁止在本模块再构造另一套模型或检查点身份。
    from .data import episode_messages
    from .tokenize import tokenize_trajectory, TrajectoryTooLong
    raw=[json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    selected=[r for r in raw if (task_id is None or r.get('task_id')==task_id)
              and (repeat is None or r.get('repeat',0)==repeat)]
    if not selected:
        raise ValueError('原始轨迹中没有匹配的任务/repeat')
    rows=[];rejected=[];seen=set()
    for row in selected:
        task=row.get('task_id')
        if type(task) is not int or task in seen:
            raise ValueError('原始轨迹任务重复或ID无效；请用--task-id/--repeat选择唯一轨迹')
        seen.add(task)
        try:
            encoded=tokenize_trajectory(tokenizer,episode_messages(row,require_labels=False),max_length)
        except TrajectoryTooLong as error:
            rejected.append({'task_id':task,'repeat':row.get('repeat',0),'reason':str(error)})
            continue
        rows.append((task,encoded))
    return rows,rejected,len(selected)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','checkpoint','output'):
        p.add_argument('--'+key,required=True,type=Path)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--samples',type=Path)
    source.add_argument('--rollout',type=Path)
    p.add_argument('--task-id',type=int)
    p.add_argument('--repeat',type=int)
    p.add_argument('--source-commit',help='可选断言实际runtime源码提交；不能用来改写checkpoint身份')
    p.add_argument('--reference',type=Path)
    p.add_argument('--atol',type=float,default=1e-5)
    p.add_argument('--rtol',type=float,default=1e-5)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    args=p.parse_args(argv)
    if args.samples and any(v is not None for v in (args.task_id,args.repeat)):
        raise ValueError('--task-id/--repeat仅用于原始rollout输入')
    if args.output.exists() or args.checkpoint.resolve() in args.output.resolve().parents:
        raise ValueError('新输出须位于检查点之外，不能覆盖旧证据')
    cfg=load_config(args.config)
    model,tok,identity,inputs=build_prm(cfg,args.checkpoint,args.device,False)
    if args.source_commit is not None and args.source_commit!=identity['source_commit']:
        raise ValueError('声明的源码提交与当前runtime不符')
    if args.samples:
        rows,rejected,total=load_samples(args.samples,tok,cfg.max_length,training=False)
    else:
        rows,rejected,total=load_rollout_samples(args.rollout,tok,cfg.max_length,args.task_id,args.repeat)
    if not rows:
        raise ValueError('无可评分的完整轨迹')
    actual=validate_predictions(predict(model,tok,rows))
    if args.reference:
        if rejected:
            raise ValueError('重载验收输入有剔除项，不能只比较保留子集')
        compare_predictions(actual,json.loads(args.reference.read_text(encoding='utf-8')),args.atol,args.rtol)
    else:
        # 无reference仍检查容差是否合法，不能把非法值写入证据。
        compare_predictions(actual,actual,args.atol,args.rtol)
    if verify_runtime_inputs(cfg)!=inputs:
        raise ValueError('推理过程中初始化文件改变')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,{'predictions':actual,'identity':identity,'device':args.device,
                          'total_samples':total,'rejected':rejected,
                          'input_schema':'samples' if args.samples else 'rollout',
                          'reference_match':True if args.reference else None,
                          'atol':args.atol,'rtol':args.rtol,'test_authorization':False})


if __name__=='__main__':
    main()
