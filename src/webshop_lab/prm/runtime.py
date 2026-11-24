"""接入固定Qwen配置、独立PRM LoRA与完整推理检查点。"""
import hashlib
import json
import subprocess
from pathlib import Path
import torch
from .config import verify_input_files
from .tokenize import collate_trajectories
from .model_core import NeuralPRM, restore_head, verify_prm


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def verify_runtime_inputs(config):
    observed=verify_input_files(config)
    # 加载行为还受这些辅助文件影响，不能只记录主权重。
    for root,names in ((Path(config.model),('config.json','generation_config.json')),
                       (Path(config.tokenizer),('special_tokens_map.json','added_tokens.json','chat_template.jinja'))):
        for name in names:
            path=root/name
            if path.is_file():
                observed[str(path)]=sha(path)
    return observed


def tensor_batch(examples, pad_token_id, device, training):
    """只转发模型接收的字段；标签与assistant旁路mask不混入token输入。"""
    raw = collate_trajectories(examples, pad_token_id)
    result = {key: torch.tensor(raw[key], dtype=dtype, device=device) for key,dtype in (
        ('input_ids',torch.long), ('attention_mask',torch.long),
        ('turn_end_indices',torch.long), ('turn_mask',torch.bool))}
    if training:
        if 'terminal_reward' not in raw:
            raise ValueError('训练批次缺少终局奖励标签')
        result['labels'] = torch.tensor(raw['terminal_reward'], dtype=torch.float32, device=device)
    return result


def checkpoint_identity(config, tokenizer, inputs, device):
    """训练与所有推理入口共用的严格身份；不接收外部伪造的source或manifest。"""
    source = {name: sha(Path(__file__).with_name(name)) for name in
              ('runtime.py','model_core.py','train_core.py','tokenize.py','config.py')}
    git = subprocess.run(['git','rev-parse','HEAD'], cwd=Path(__file__).parent,
                         capture_output=True,text=True)
    status = subprocess.run(['git','status','--porcelain','--untracked-files=no'],
                            cwd=Path(__file__).parent,capture_output=True,text=True)
    tracked = subprocess.run(['git','ls-files','--error-unmatch',*source],
                             cwd=Path(__file__).parent,capture_output=True,text=True)
    clean = git.returncode == status.returncode == tracked.returncode == 0 and not status.stdout.strip()
    if device == 'cuda' and not clean:
        raise ValueError('目标GPU须在包含全部PRM源码的干净Git检出执行')
    return {
        'base_revision':config.revision, 'base_sha256':config.base_weights_sha256,
        'tokenizer_sha256':config.tokenizer_files_sha256,
        'tokenizer_revision':config.tokenizer_revision,
        'auxiliary_input_sha256':{path:digest for path,digest in inputs.items()
                                 if Path(path).name in ('config.json','generation_config.json',
                                     'special_tokens_map.json','added_tokens.json','chat_template.jinja')},
        'template_sha256':hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        'initialization':{'kind':config.initialization,
                          'sft_adapter_files_sha256':config.sft_adapter_files_sha256},
        'source_commit':git.stdout.strip() if git.returncode == 0 else '未记录Git提交（仅结构测试）',
        'source_tracked_clean':clean,
        'source_sha256':source,
        'prm_lora':{'rank':config.lora_r,'alpha':config.lora_alpha,
                    'target_modules':list(config.target_modules),'dropout':0.05},
        'score_position':config.score_position, 'max_length':config.max_length,
        'policy_dtype':'float32'}


def build_prm(config, checkpoint=None, device='cuda', training=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    if config.policy_dtype != 'float32' or config.load_in_4bit:
        raise ValueError('本设备契约仅允许非量化FP32')
    if config.max_length > 2048:
        raise ValueError('本次上下文上限为2048，不自动扩大资源契约')
    if device not in ('cpu','cuda') or (device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('设备不可用；CPU只作独立结构测试，不代表目标GPU验收')
    if not Path(config.model).is_dir() or not Path(config.tokenizer).is_dir():
        raise ValueError('仅允许已核对的本地模型与tokenizer目录')
    before = verify_runtime_inputs(config)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.is_autocast_enabled():
        raise ValueError('运行前必须关闭autocast')
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer,
        revision=config.tokenizer_revision, local_files_only=True,
        trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    identity = checkpoint_identity(config,tokenizer,before,device)
    if checkpoint is not None:
        # Reject a wrong representation/schema before allocating the 1.5B backbone.
        model_spec=json.loads((Path(config.model)/'config.json').read_text(encoding='utf-8'))
        vocab_size,hidden_size=model_spec.get('vocab_size'),model_spec.get('hidden_size')
        if type(vocab_size) is not int or type(hidden_size) is not int:
            raise ValueError('Pinned model configuration lacks exact feature dimensions')
        verify_prm(checkpoint,identity,config.representation,vocab_size,hidden_size)
    base = AutoModelForCausalLM.from_pretrained(config.model, revision=config.revision,
        local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32,
        attn_implementation='sdpa')
    if config.initialization == 'sft_b1':
        # 先保留SFT初始化，再建立新的PRM LoRA；合并只在内存中发生。
        base = PeftModel.from_pretrained(base,config.sft_adapter,is_trainable=False,
                                         local_files_only=True).merge_and_unload(safe_merge=True)
        if any('lora_' in name for name,_ in base.named_parameters()):
            raise ValueError('B1合并后仍残留adapter参数，不允许重复叠加')
        # PEFT 0.17.1卸载层后保留配置属性；清除失效元数据再挂新PRM。
        if hasattr(base,'peft_config'):
            delattr(base,'peft_config')
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    base.config.use_cache = False
    if checkpoint is None:
        backbone = get_peft_model(base,LoraConfig(r=config.lora_r,
            lora_alpha=config.lora_alpha,target_modules=list(config.target_modules),
            lora_dropout=0.05,bias='none',task_type='CAUSAL_LM'))
    else:
        backbone = PeftModel.from_pretrained(base,str(Path(checkpoint)/'adapter'),
            is_trainable=training,local_files_only=True)
    checkpoint_schema=2 if config.schema_version=='prm-qwen-config-v2' else 1
    model = NeuralPRM(backbone,representation=config.representation,
                      checkpoint_schema_version=checkpoint_schema).to(device)
    if checkpoint is not None:
        restore_head(model,checkpoint,identity)
    if training:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        backbone.enable_input_require_grads()
        names = [n for n,p in model.named_parameters() if p.requires_grad]
        if not names or any('lora_' not in n and not n.startswith('score_head.') for n in names):
            raise ValueError('只有PRM LoRA和评分头允许训练')
    else:
        model.requires_grad_(False)
        model.eval()
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('实际模型参数不全为FP32')
    if verify_runtime_inputs(config) != before:
        raise ValueError('初始化过程中输入文件变化')
    return model,tokenizer,identity,before


def finalize_manifest(directory):
    """把运行配置、训练状态及参考分数一并纳入清单。"""
    directory = Path(directory)
    manifest = json.loads((directory/'prm-manifest.json').read_text(encoding='utf-8'))
    files = {str(p.relative_to(directory)):sha(p) for p in sorted(directory.rglob('*'))
             if p.is_file() and p.name not in ('prm-manifest.json','SHA256SUMS')}
    manifest['files_sha256'] = files
    write_json(directory/'prm-manifest.json',manifest)
    files['prm-manifest.json'] = sha(directory/'prm-manifest.json')
    (directory/'SHA256SUMS').write_text(''.join(f'{h}  {p}\n' for p,h in sorted(files.items())),encoding='utf-8')
