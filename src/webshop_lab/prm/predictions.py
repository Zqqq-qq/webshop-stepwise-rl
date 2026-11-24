"""所有PRM重载入口共用的完整性、有限数值与容差校验；不依赖Torch。"""
import math


def validate_predictions(rows):
    if not isinstance(rows,list) or not rows:
        raise ValueError('预测必须是非空列表，不能用空比较通过验收')
    tasks=set()
    for row in rows:
        if not isinstance(row,dict) or type(row.get('task_id')) is not int:
            raise ValueError('预测缺少整数任务ID')
        if row['task_id'] in tasks:
            raise ValueError('预测任务ID重复；重复采样须先按repeat单独选择')
        tasks.add(row['task_id'])
        ends=row.get('turn_end_indices');scores=row.get('step_scores')
        if (not isinstance(ends,list) or not ends or
                any(type(i) is not int or i<0 for i in ends) or
                any(b<=a for a,b in zip(ends,ends[1:])) or
                not isinstance(scores,list) or len(scores)!=len(ends)):
            raise ValueError('评分位置与逐步分数须完整、非空、长度相等且严格递增')
        for value in scores+[row.get('total')]:
            if type(value) not in (int,float) or not math.isfinite(value):
                raise ValueError('逐步分数及总分必须为有限数值')
    return rows


def compare_predictions(actual,expected,atol,rtol):
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in (atol,rtol)):
        raise ValueError('容差必须为非负有限数')
    validate_predictions(actual);validate_predictions(expected)
    if len(actual)!=len(expected):
        raise ValueError('重载样本数不一致')
    for a,b in zip(actual,expected):
        if a['task_id']!=b['task_id'] or a['turn_end_indices']!=b['turn_end_indices']:
            raise ValueError('重载任务顺序或动作位置不一致')
        for x,y in zip(a['step_scores']+[a['total']],b['step_scores']+[b['total']]):
            if not math.isclose(x,y,abs_tol=atol,rel_tol=rtol):
                raise ValueError('独立进程逐步或总分超出预定容差')
