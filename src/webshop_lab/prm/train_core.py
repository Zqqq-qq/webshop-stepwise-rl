"""最小真实优化步骤；模板/数据/初始化由适配入口提供。"""
import torch


def optimizer_step(model, batch, optimizer, max_grad_norm=1.0):
    return accumulated_optimizer_step(model, (batch,), optimizer, max_grad_norm)


def accumulated_optimizer_step(model, batches, optimizer, max_grad_norm=1.0):
    """Average trajectory losses, then clip once and take one optimizer step."""
    batches = tuple(batches)
    sizes = [batch['labels'].numel() for batch in batches]
    if not sizes or min(sizes) < 1:
        raise ValueError('每次优化至少需要一条带标签的完整轨迹')
    trajectories = sum(sizes)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    mean_loss = 0.
    totals = []
    for batch, size in zip(batches, sizes):
        output = model(**batch)
        loss = output['loss']
        if loss is None or not bool(torch.isfinite(loss)):
            raise ValueError('缺少有限PRM损失')
        # Each model loss is already a mean over the microbatch's trajectories.
        # Backward now frees its activation graph before the next forward.
        weight = size / trajectories
        (loss * weight).backward()
        mean_loss += float(loss.detach()) * weight
        totals.extend(output['totals'].detach().cpu().tolist())
        del output, loss
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not any(p.grad is not None for p in parameters):
        raise ValueError('PRM梯度路径被切断')
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return {'loss': mean_loss, 'gradient_norm': float(norm), 'totals_before_update': totals}
