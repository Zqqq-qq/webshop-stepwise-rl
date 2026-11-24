"""原PRM的末位词表logits评分定义；不负责模板和数据转换。"""
import hashlib
import json
from pathlib import Path
import torch
from torch import nn


class NeuralPRM(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.vocab_size = int(backbone.config.vocab_size)
        if self.vocab_size <= 0:
            raise ValueError('词表维度必须为正')
        parameter = next(backbone.parameters())
        self.score_head = nn.Linear(self.vocab_size, 1, device=parameter.device,
                                    dtype=parameter.dtype)
        nn.init.zeros_(self.score_head.weight)
        nn.init.zeros_(self.score_head.bias)

    def forward(self, input_ids, attention_mask, turn_end_indices, turn_mask, labels=None):
        if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
            raise ValueError('输入与attention_mask必须为相同的二维形状')
        if turn_end_indices.ndim != 2 or turn_end_indices.shape != turn_mask.shape:
            raise ValueError('评分位置与turn_mask必须为相同的二维形状')
        if turn_end_indices.dtype != torch.long or turn_mask.dtype != torch.bool:
            raise ValueError('评分位置须为long，turn_mask须为bool')
        batch, length = input_ids.shape
        if turn_mask.shape[0] != batch or not bool(turn_mask.any(dim=1).all()):
            raise ValueError('每条轨迹至少需要一个真实assistant评分位置')
        if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            raise ValueError('attention_mask必须为0或1')
        if not bool((turn_end_indices[~turn_mask] == -1).all()):
            raise ValueError('无效步骤的位置统一使用-1')
        for i in range(batch):
            ends = turn_end_indices[i][turn_mask[i]]
            if not bool(((ends >= 0) & (ends < length)).all()):
                raise ValueError('评分位置越界')
            if not bool((attention_mask[i, ends] == 1).all()):
                raise ValueError('padding不能作为评分位置')
            if ends.numel() > 1 and not bool((ends[1:] > ends[:-1]).all()):
                raise ValueError('评分位置必须严格递增且无重复')
        # 参考实现会物化[B,L,V]。取位在其后，不宣称减少logits峰值显存。
        logits = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                               use_cache=False).logits
        if logits.shape != (batch, length, self.vocab_size):
            raise ValueError('实际logits维度与模型词表配置不一致')
        safe = turn_end_indices.clamp_min(0)
        selected = logits[torch.arange(batch, device=logits.device)[:, None], safe]
        step_scores = self.score_head(selected.to(self.score_head.weight.dtype)).squeeze(-1)
        step_scores = step_scores.masked_fill(~turn_mask, 0)
        totals = step_scores.float().sum(dim=1)
        loss = None
        if labels is not None:
            if labels.shape != (batch,) or not bool(torch.isfinite(labels).all()):
                raise ValueError('终局标签必须为每轨迹一个有限标量')
            if not bool(((labels >= 0) & (labels <= 1)).all()):
                raise ValueError('终局任务奖励必须位于[0,1]')
            loss = (totals - labels.float()).square().mean()
        return {'loss': loss, 'step_scores': step_scores, 'totals': totals}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_prm(model, directory, identity):
    """保存PRM专用adapter和独立评分头；调用方须提供完整初始化身份。"""
    directory = Path(directory)
    required = {'base_revision', 'base_sha256', 'tokenizer_sha256',
                'initialization', 'template_sha256', 'source_commit'}
    if not required <= identity.keys() or any(not identity[k] for k in required):
        raise ValueError('缺少基础模型、模板、tokenizer或初始化来源身份')
    if directory.exists():
        raise ValueError('拒绝覆盖已有PRM检查点')
    directory.mkdir(parents=True)
    model.backbone.save_pretrained(directory / 'adapter')
    torch.save(model.score_head.state_dict(), directory / 'head.pt')
    files = {str(p.relative_to(directory)): _sha(p)
             for p in sorted(directory.rglob('*')) if p.is_file()}
    if not any(p.startswith('adapter/') and p.endswith('.safetensors') for p in files):
        raise ValueError('未保存PRM adapter权重')
    manifest = {'schema_version': 1, 'identity': identity, 'vocab_size': model.vocab_size,
                'score_definition': 'assistant_turn_end_vocab_logits_linear_sum_mse',
                'files_sha256': files}
    (directory / 'prm-manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def verify_prm(directory, expected_identity):
    """必须先验身份再由集成入口加载adapter，禁止只加载head冒充完整重载。"""
    directory = Path(directory)
    manifest = json.loads((directory/'prm-manifest.json').read_text(encoding='utf-8'))
    if manifest['schema_version'] != 1 or manifest['identity'] != expected_identity:
        raise ValueError('PRM初始化或输入身份不匹配')
    if manifest['score_definition'] != 'assistant_turn_end_vocab_logits_linear_sum_mse':
        raise ValueError('PRM评分定义不匹配')
    files = manifest['files_sha256']
    if 'head.pt' not in files or not any(p.startswith('adapter/') and p.endswith('.safetensors') for p in files):
        raise ValueError('检查点缺少adapter或评分头')
    for name, digest in files.items():
        p = (directory/name).resolve()
        if not p.is_relative_to(directory.resolve()) or _sha(p) != digest:
            raise ValueError('PRM检查点路径或文件哈希错误')
    return manifest


def restore_head(model, directory, expected_identity):
    """调用方先恢复基础初始化与PRM adapter，再恢复此head。"""
    manifest = verify_prm(directory, expected_identity)
    if manifest['vocab_size'] != model.vocab_size:
        raise ValueError('重载词表维度不一致')
    state = torch.load(Path(directory)/'head.pt', map_location='cpu', weights_only=True)
    if not all(bool(torch.isfinite(v).all()) for v in state.values()):
        raise ValueError('评分头包含非有限权重')
    model.score_head.load_state_dict(state, strict=True)
