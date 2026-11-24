"""CPU-only, versioned configuration for Qwen neural PRM initialization."""
from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
from pathlib import Path
import re


def _digest(value, name):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
        raise ValueError(f'{name} must be a lowercase SHA-256')


def _integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')


@dataclass(frozen=True)
class PRMConfig:
    model: str
    revision: str
    base_weights_sha256: str
    tokenizer: str
    tokenizer_revision: str
    tokenizer_files_sha256: dict
    output_dir: str
    initialization: str = 'base'
    sft_adapter: str | None = None
    sft_adapter_files_sha256: dict = field(default_factory=dict)
    policy_dtype: str = 'float32'
    load_in_4bit: bool = False
    max_length: int = 2048
    lora_r: int = 8
    lora_alpha: int = 16
    target_modules: tuple = ('q_proj', 'v_proj')
    schema_version: str = 'prm-qwen-config-v1'
    score_position: str = 'assistant_im_end'

    def __post_init__(self):
        if self.schema_version != 'prm-qwen-config-v1' or self.score_position != 'assistant_im_end':
            raise ValueError('Unsupported PRM schema or score-position convention')
        for name in ('model', 'tokenizer', 'output_dir'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f'{name} must be a nonempty string')
        for name in ('revision', 'tokenizer_revision'):
            if not isinstance(getattr(self, name), str) or not re.fullmatch(r'[0-9a-f]{40}', getattr(self, name)):
                raise ValueError(f'{name} must be an exact lowercase commit SHA')
        _digest(self.base_weights_sha256, 'base_weights_sha256')
        if self.initialization not in ('base', 'sft_b1'):
            raise ValueError('initialization must be base or sft_b1')
        if not isinstance(self.tokenizer_files_sha256, dict) or not self.tokenizer_files_sha256:
            raise ValueError('Record tokenizer file identities')
        for mapping in (self.tokenizer_files_sha256, self.sft_adapter_files_sha256):
            if not isinstance(mapping, dict):
                raise ValueError('File identities must be dictionaries')
            for name, digest in mapping.items():
                if not isinstance(name, str) or not name or '/' in name or '\\' in name or name in ('.', '..'):
                    raise ValueError('Identity keys must be plain file names')
                _digest(digest, name)
        if self.initialization == 'sft_b1':
            if not isinstance(self.sft_adapter, str) or not self.sft_adapter.strip():
                raise ValueError('sft_b1 initialization requires the original adapter path')
            if set(self.sft_adapter_files_sha256) != {'adapter_model.safetensors', 'adapter_config.json'}:
                raise ValueError('Record both original B1 adapter file hashes')
        elif self.sft_adapter is not None or self.sft_adapter_files_sha256:
            raise ValueError('base initialization cannot silently load an SFT adapter')
        if self.policy_dtype not in ('float32', 'bfloat16', 'float16') or type(self.load_in_4bit) is not bool:
            raise ValueError('Invalid dtype or quantization flag')
        if self.load_in_4bit and self.policy_dtype == 'float32':
            raise ValueError('A 4-bit backbone cannot be described as a float32 policy')
        for name in ('max_length', 'lora_r', 'lora_alpha'):
            _integer(getattr(self, name), name)
        if not isinstance(self.target_modules, (tuple, list)) or not self.target_modules:
            raise ValueError('target_modules must be a nonempty list/tuple')
        if any(not isinstance(m, str) or not m.strip() for m in self.target_modules):
            raise ValueError('Invalid LoRA module name')
        if len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError('Duplicate LoRA target modules')
        object.__setattr__(self, 'target_modules', tuple(self.target_modules))
        destinations = {str(Path(self.model).expanduser().resolve())}
        if self.sft_adapter:
            destinations.add(str(Path(self.sft_adapter).expanduser().resolve()))
        output = Path(self.output_dir).expanduser().resolve()
        if any(output == Path(p) or Path(p) in output.parents for p in destinations):
            raise ValueError('PRM output must be separate from base/B1 input directories')

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError('PRM configuration must be an object')
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f'Unknown PRM configuration fields: {sorted(unknown)}')
        try:
            return cls(**data)
        except TypeError as exc:
            raise ValueError(f'Invalid PRM configuration: {exc}') from exc

    def to_dict(self):
        result = asdict(self)
        result['target_modules'] = list(self.target_modules)
        return result

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_config(path):
    return PRMConfig.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))


def verify_input_files(config):
    """Verify local payload identities before model load; never import Torch."""
    expected = [(Path(config.model) / 'model.safetensors', config.base_weights_sha256)]
    expected += [(Path(config.tokenizer) / name, digest)
                 for name, digest in config.tokenizer_files_sha256.items()]
    expected += [(Path(config.sft_adapter) / name, digest)
                 for name, digest in config.sft_adapter_files_sha256.items()]
    observed = {}
    for path, digest in expected:
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(block)
        if h.hexdigest() != digest:
            raise ValueError(f'PRM input hash mismatch: {path.name}')
        observed[str(path)] = h.hexdigest()
    return observed
