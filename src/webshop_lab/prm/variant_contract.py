"""Issue48变体的待实施身份契约；不接入现行runtime，不加载/迁移检查点。"""
from dataclasses import asdict,dataclass

LEGACY_SCORE='assistant_turn_end_vocab_logits_linear_sum_mse'


@dataclass(frozen=True)
class RepresentationContract:
    representation: str
    input_dim: int
    head_type: str='linear'
    score_position: str='assistant_im_end'
    reduction: str='sum'
    loss: str='trajectory_mse'
    dtype: str='float32'
    initialization: str='zeros'

    def __post_init__(self):
        if self.representation not in ('vocab_logits','final_hidden_state'):
            raise ValueError('未知特征表示')
        if type(self.input_dim) is not int or self.input_dim<=0:
            raise ValueError('输入维度须为正整数')
        expected={'head_type':'linear','score_position':'assistant_im_end','reduction':'sum',
                  'loss':'trajectory_mse','dtype':'float32','initialization':'zeros'}
        if any(getattr(self,k)!=v for k,v in expected.items()):
            raise ValueError('首轮配对方案不能同时改变头类型、位置、聚合、损失、精度或初始化')

    def validate_dimensions(self,vocab_size,hidden_size):
        if any(type(v) is not int or v<=0 for v in (vocab_size,hidden_size)):
            raise ValueError('实际模型维度无效')
        expected=vocab_size if self.representation=='vocab_logits' else hidden_size
        if self.input_dim!=expected:
            raise ValueError('评分头维度与实际表示不一致')

    @property
    def score_definition(self):
        if self.representation=='vocab_logits':return LEGACY_SCORE
        return 'assistant_im_end_final_norm_hidden_linear_sum_mse'

    def manifest_header(self,identity,vocab_size,hidden_size):
        """仅生成设计中的v2头部；文件SHA和实际tensor检查仍由未来完整loader负责。"""
        self.validate_dimensions(vocab_size,hidden_size)
        return {'schema_version':2,'identity':identity,'architecture':asdict(self),
                'vocab_size':vocab_size,'hidden_size':hidden_size,
                'score_definition':self.score_definition}


def check_manifest_header(manifest,expected,identity,vocab_size,hidden_size):
    """仅检查类型/维度/身份头部，绝不将此函数当作完整文件或重载验收。"""
    expected.validate_dimensions(vocab_size,hidden_size)
    if manifest.get('identity')!=identity or manifest.get('vocab_size')!=vocab_size:
        raise ValueError('初始化身份或词表不一致')
    if manifest.get('schema_version')==1:
        if expected.representation!='vocab_logits' or manifest.get('score_definition')!=LEGACY_SCORE:
            raise ValueError('旧v1检查点不能冒充hidden头')
        return 'legacy_v1_requires_pinned_runtime'
    if manifest!=expected.manifest_header(identity,vocab_size,hidden_size):
        # 调用者仅传入头部；未来完整manifest会另分离files_sha256再核文件。
        raise ValueError('v2表示、维度或评分定义不匹配')
    return 'v2_header_only_not_full_reload'
