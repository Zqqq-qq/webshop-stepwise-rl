"""最小真实优化步骤；模板/数据/初始化由适配入口提供。"""
import torch


def optimizer_step(model, batch, optimizer, max_grad_norm=1.0):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(**batch)
    loss = output['loss']
    if loss is None or not bool(torch.isfinite(loss)):
        raise ValueError('缺少有限PRM损失')
    loss.backward()
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not any(p.grad is not None for p in parameters):
        raise ValueError('PRM梯度路径被切断')
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return {'loss': float(loss.detach()), 'gradient_norm': float(norm),
            'totals_before_update': output['totals'].detach().cpu().tolist()}
