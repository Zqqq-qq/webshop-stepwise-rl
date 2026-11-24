"""Qwen text-only trajectory encoding. No Torch, model weights or task labels."""
import hashlib
import math


class TrajectoryTooLong(ValueError):
    """A complete trajectory cannot fit; caller must count/report this rejection."""


def tokenize_trajectory(tokenizer, messages, max_length):
    """Score each assistant at its im_end token, not at next-token prediction offset.

    Spans are half-open token intervals containing assistant content plus im_end.
    A BPE token merging leading content whitespace with the header's final newline
    belongs to that span. Role names, other messages and trailing newlines do not.
    Uses offsets from ONE complete encoding, never concatenated per-message IDs.
    """
    if type(max_length) is not int or max_length < 1:
        raise ValueError('max_length must be a positive integer')
    if not getattr(tokenizer, 'is_fast', False) or not getattr(tokenizer, 'chat_template', None):
        raise ValueError('A fast tokenizer with its recorded chat template is required')
    if not isinstance(messages, list) or not messages:
        raise ValueError('Supply complete text messages')
    special = set(tokenizer.all_special_tokens)
    clean = []
    for message in messages:
        if not isinstance(message, dict) or set(message) != {'role', 'content'}:
            raise ValueError('Only explicit role/content text messages are supported')
        role, content = message['role'], message['content']
        if role not in ('system', 'user', 'assistant') or not isinstance(content, str):
            raise ValueError('Unsupported message role/content')
        if any(token in content for token in special):
            raise ValueError('Reserved template token in message content')
        if role == 'assistant' and not content.strip():
            raise ValueError('Empty assistant action cannot be scored')
        clean.append(dict(message))
    if clean[0]['role'] != 'system' or clean[-1]['role'] != 'assistant':
        raise ValueError('Supply explicit system and end at the last assistant action')
    if any(m['role'] == 'system' for m in clean[1:]):
        raise ValueError('Only the initial system message is supported')
    render = lambda rows, generation=False: tokenizer.apply_chat_template(
        rows, tokenize=False, add_generation_prompt=generation)
    text = render(clean)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False, padding=False)
    ids, offsets = list(encoded['input_ids']), encoded['offset_mapping']
    if len(ids) > max_length:
        raise TrajectoryTooLong(f'Complete trajectory has {len(ids)} tokens; limit={max_length}')
    template_ids = tokenizer.apply_chat_template(clean, tokenize=True, add_generation_prompt=False)
    if ids != list(template_ids):
        raise ValueError('Rendered and template token IDs disagree')
    eos = tokenizer.convert_tokens_to_ids('<|im_end|>')
    if eos is None or tokenizer.eos_token_id != eos or tokenizer.decode([eos]) != '<|im_end|>':
        raise ValueError('Unsupported template: expected Qwen im_end EOS')
    spans, ends, mask = [], [], [0] * len(ids)
    for index, message in enumerate(clean):
        if message['role'] != 'assistant':
            continue
        prefix = render(clean[:index], True)
        completed = render(clean[:index + 1])
        expected = prefix + message['content'] + '<|im_end|>\n'
        if completed != expected or not text.startswith(completed):
            raise ValueError('Template is not a prefix-stable plain Qwen conversation')
        begin_char = len(prefix)
        end_char = len(completed) - 1  # exclude trailing newline; include im_end
        positions = [i for i, (a, b) in enumerate(offsets) if b > begin_char and a < end_char]
        if not positions:
            raise ValueError('Assistant span is empty')
        start, end = positions[0], positions[-1] + 1
        # Only whitespace at the header boundary may merge with content in a BPE token.
        a = offsets[start][0]
        if a < begin_char and text[a:begin_char].strip():
            raise ValueError('Token crosses an assistant role boundary')
        if ids[end - 1] != eos or offsets[end - 1][1] != end_char:
            raise ValueError('Assistant scoring token is not its exact end marker')
        if any(mask[start:end]):
            raise ValueError('Overlapping assistant spans')
        spans.append([start, end])
        ends.append(end - 1)
        mask[start:end] = [1] * (end - start)
    if not ends:
        raise ValueError('No assistant turns')
    return {'input_ids': ids, 'attention_mask': [1] * len(ids), 'assistant_mask': mask,
            'assistant_spans': spans, 'turn_end_indices': ends, 'turn_mask': [1] * len(ends),
            'template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
            'score_position': 'assistant_im_end'}


def collate_trajectories(examples, pad_token_id):
    """Right-pad Python lists; model owner converts these to device tensors.

    Invalid turn indices use -1 and MUST be removed using turn_mask before gather.
    Optional terminal_reward is a separate [batch] label, never appended to inputs.
    """
    if not examples or type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError('Nonempty examples and a valid padding ID are required')
    for row in examples:
        ids, attention, mask = row['input_ids'], row['attention_mask'], row['assistant_mask']
        spans, ends, turns = row['assistant_spans'], row['turn_end_indices'], row['turn_mask']
        if not ids or len(ids) != len(attention) or len(ids) != len(mask) or any(v != 1 for v in attention):
            raise ValueError('Collate expects unpadded tokenized trajectories')
        if not ends or len(spans) != len(ends) or turns != [1] * len(ends):
            raise ValueError('Invalid turn alignment')
        expected, previous = [0] * len(ids), 0
        for (start, end), at in zip(spans, ends):
            if not (type(start) is int and type(end) is int and previous <= start < end <= len(ids) and at == end - 1):
                raise ValueError('Invalid assistant span or scoring index')
            expected[start:end] = [1] * (end - start)
            previous = end
        if mask != expected:
            raise ValueError('Assistant mask differs from spans')
        if row.get('score_position') != 'assistant_im_end':
            raise ValueError('Mixed scoring conventions')
    templates = {row['template_sha256'] for row in examples}
    if len(templates) != 1:
        raise ValueError('Mixed tokenizer templates')
    width = max(len(row['input_ids']) for row in examples)
    count = max(len(row['turn_end_indices']) for row in examples)
    result = {key: [] for key in ('input_ids', 'attention_mask', 'assistant_mask', 'turn_end_indices', 'turn_mask')}
    for row in examples:
        pad, extra = width - len(row['input_ids']), count - len(row['turn_end_indices'])
        result['input_ids'].append(row['input_ids'] + [pad_token_id] * pad)
        result['attention_mask'].append(row['attention_mask'] + [0] * pad)
        result['assistant_mask'].append(row['assistant_mask'] + [0] * pad)
        result['turn_end_indices'].append(row['turn_end_indices'] + [-1] * extra)
        result['turn_mask'].append(row['turn_mask'] + [0] * extra)
    has_labels = ['terminal_reward' in row for row in examples]
    if any(has_labels):
        if not all(has_labels):
            raise ValueError('Mixed labeled and unlabeled samples')
        labels = [row['terminal_reward'] for row in examples]
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in labels):
            raise ValueError('Invalid terminal reward labels')
        result['terminal_reward'] = labels
    return result
