"""原PRM的末位词表logits评分定义；不负责模板和数据转换。"""
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from .variant_contract import RepresentationContract, check_manifest_header


class NeuralPRM(nn.Module):
    def __init__(self, backbone, representation='vocab_logits', checkpoint_schema_version=1):
        super().__init__()
        self.backbone = backbone
        self.vocab_size = int(backbone.config.vocab_size)
        if self.vocab_size <= 0:
            raise ValueError('词表维度必须为正')
        self.hidden_size = getattr(backbone.config, 'hidden_size', None)
        if representation == 'final_hidden_state':
            if type(self.hidden_size) is not int or self.hidden_size <= 0:
                raise ValueError('Hidden-state PRM requires a positive model hidden_size')
            if not callable(getattr(backbone, 'get_base_model', None)):
                raise ValueError('Hidden-state PRM requires an explicit PEFT base model')
        input_dim = self.hidden_size if representation == 'final_hidden_state' else self.vocab_size
        self.contract = RepresentationContract(representation, input_dim)
        self.representation = representation
        if checkpoint_schema_version not in (1, 2):
            raise ValueError('Unsupported PRM checkpoint schema')
        if checkpoint_schema_version == 1 and representation != 'vocab_logits':
            raise ValueError('Legacy checkpoint schema cannot store hidden-state heads')
        if checkpoint_schema_version == 2 and (type(self.hidden_size) is not int or self.hidden_size <= 0):
            raise ValueError('v2 checkpoint requires a known backbone hidden_size')
        self.checkpoint_schema_version = checkpoint_schema_version
        parameter = next(backbone.parameters())
        self.score_head = nn.Linear(input_dim, 1, device=parameter.device,
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
        if self.representation == 'vocab_logits':
            # Legacy definition: explicitly materialize [B,L,V], with unchanged scoring.
            features = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                                     use_cache=False).logits
            feature_dim = self.vocab_size
        else:
            # The same causal Qwen decoder and active PEFT LoRA, stopping after final RMSNorm.
            # Do not call the LM head or collect hidden_states of all transformer layers.
            base = self.backbone.get_base_model()
            decoder = getattr(base, 'model', None)
            if decoder is None:
                raise ValueError('Hidden-state PRM requires Qwen base.model decoder')
            features = decoder(input_ids=input_ids, attention_mask=attention_mask,
                               use_cache=False, return_dict=True).last_hidden_state
            feature_dim = self.hidden_size
        if features.shape != (batch, length, feature_dim):
            raise ValueError('Actual score feature shape does not match the model contract')
        safe = turn_end_indices.clamp_min(0)
        selected = features[torch.arange(batch, device=features.device)[:, None], safe]
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
    if model.checkpoint_schema_version == 1:
        manifest = {'schema_version': 1, 'identity': identity, 'vocab_size': model.vocab_size,
                    'score_definition': 'assistant_turn_end_vocab_logits_linear_sum_mse',
                    'files_sha256': files}
    else:
        manifest = model.contract.manifest_header(identity, model.vocab_size, model.hidden_size)
        manifest['files_sha256'] = files
    (directory / 'prm-manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def verify_prm(directory, expected_identity, expected_representation='vocab_logits',
               vocab_size=None, hidden_size=None):
    """必须先验身份再由集成入口加载adapter，禁止只加载head冒充完整重载。"""
    directory = Path(directory)
    manifest = json.loads((directory/'prm-manifest.json').read_text(encoding='utf-8'))
    if manifest.get('identity') != expected_identity:
        raise ValueError('PRM初始化或输入身份不匹配')
    if manifest.get('schema_version') == 1:
        if (expected_representation != 'vocab_logits'
                or manifest.get('score_definition') != 'assistant_turn_end_vocab_logits_linear_sum_mse'):
            raise ValueError('v1 logits checkpoint cannot be loaded as another representation')
    elif manifest.get('schema_version') == 2:
        if type(vocab_size) is not int or type(hidden_size) is not int:
            raise ValueError('v2 checkpoint requires verified model dimensions')
        expected = RepresentationContract(expected_representation,
            hidden_size if expected_representation == 'final_hidden_state' else vocab_size)
        header = {k: v for k, v in manifest.items() if k != 'files_sha256'}
        check_manifest_header(header, expected, expected_identity, vocab_size, hidden_size)
    else:
        raise ValueError('Unsupported PRM checkpoint schema')
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
    manifest = verify_prm(directory, expected_identity, model.representation,
                          model.vocab_size, model.hidden_size)
    if manifest['vocab_size'] != model.vocab_size:
        raise ValueError('重载词表维度不一致')
    state = torch.load(Path(directory)/'head.pt', map_location='cpu', weights_only=True)
    if manifest['schema_version'] == 2:
        if (set(state) != {'weight', 'bias'}
                or state['weight'].shape != (1, model.contract.input_dim)
                or state['bias'].shape != (1,)
                or any(v.dtype != torch.float32 for v in state.values())):
            raise ValueError('v2 score head dimensions or dtype do not match the checkpoint contract')
    if not all(bool(torch.isfinite(v).all()) for v in state.values()):
        raise ValueError('评分头包含非有限权重')
    model.score_head.load_state_dict(state, strict=True)
