"""外置的前缀价值契约与因果编码；不替换历史检查点的固定 runtime。"""
import hashlib
import json
import math
from pathlib import Path
import re

from webshop_lab.prm.tokenize import TrajectoryTooLong, tokenize_trajectory


SEMANTICS = {
    "schema_version": "prm-prefix-value-scoring-v1",
    "head_output": "terminal_reward_prefix_value",
    "training_loss": "mean_turn_terminal_reward_mse",
    "step_reward": "value_minus_previous_value_initial_zero",
    "total": "last_prefix_value",
    "position": "assistant_im_end_per_current_prefix",
    "asin_context": "immediately_preceding_public_search_page",
    "length_policy": "current_prefix_raw_then_augmented_fallback_drop_old_pairs",
    "empty_reply": "zero_reward_keep_previous_value",
    "unscorable_prefix": "zero_reward_keep_previous_value",
}
ASIN_CLICK_RE = re.compile(r"click\[(b[0-9a-z]{9})\]", re.IGNORECASE)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def token_sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def make_contract(checkpoint):
    """由发布者显式声明 PV 语义；生成后必须独立冻结其哈希，不自动推断权重用途。"""
    manifest_path = Path(checkpoint) / "prm-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        **SEMANTICS,
        "checkpoint_manifest_sha256": file_sha(manifest_path),
        "helper_sha256": file_sha(__file__),
        "max_length": manifest["identity"]["max_length"],
        "payload_sha256": manifest["files_sha256"],
    }


def verify_contract(checkpoint, contract_path, expected_sha):
    """在分配模型之前核对外部固定契约、历史 manifest 和全部载荷。"""
    if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise ValueError("必须提供独立冻结的评分契约 SHA-256")
    if file_sha(contract_path) != expected_sha:
        raise ValueError("评分契约 SHA-256 不匹配")
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    expected_keys = set(SEMANTICS) | {"checkpoint_manifest_sha256", "helper_sha256", "max_length", "payload_sha256"}
    if set(contract) != expected_keys or any(contract.get(k) != v for k, v in SEMANTICS.items()):
        raise ValueError("不兼容的前缀价值评分语义；旧 sum 检查点不能自动迁移")
    if contract["helper_sha256"] != file_sha(__file__):
        raise ValueError("外置编码/评分契约实现哈希不匹配")
    root = Path(checkpoint).resolve()
    manifest_path = root / "prm-manifest.json"
    if file_sha(manifest_path) != contract["checkpoint_manifest_sha256"]:
        raise ValueError("评分契约绑定的检查点 manifest 不匹配")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 2
            or manifest.get("architecture", {}).get("representation") != "final_hidden_state"
            or manifest.get("identity", {}).get("max_length") != contract["max_length"]
            or type(contract["max_length"]) is not int or not 1 <= contract["max_length"] <= 2048
            or manifest.get("files_sha256") != contract["payload_sha256"]):
        raise ValueError("前缀价值契约的结构或载荷身份不匹配")
    for name, digest in contract["payload_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or file_sha(path) != digest:
            raise ValueError("评分契约载荷路径或哈希错误")
    return contract


def extract_preclick_asin_snippet(messages, msg_idx):
    if (msg_idx <= 0 or messages[msg_idx].get("role") != "assistant"
            or messages[msg_idx - 1].get("role") != "user"):
        return None
    match = ASIN_CLICK_RE.search(messages[msg_idx].get("content", ""))
    if not match:
        return None
    asin_raw = match.group(1)
    page = messages[msg_idx - 1].get("content", "")
    pattern = re.compile(r"\[SEP\]\s*" + re.escape(asin_raw.upper())
                         + r"\s*\[SEP\]\s*(.*?)\s*\[SEP\]\s*(\$[^\[]+?)(?:\s*\[SEP\]|$)", re.DOTALL)
    hit = pattern.search(page)
    if not hit:
        return None
    title, price = " ".join(hit.group(1).split())[:180], " ".join(hit.group(2).split())[:32]
    content = messages[msg_idx]["content"]
    return {"assistant_message_index": msg_idx, "source_search_message_index": msg_idx - 1,
            "clicked_asin_raw": asin_raw, "matched_asin_upper": asin_raw.upper(),
            "source_char_span": [hit.start(), hit.end()], "source_verbatim_record": page[hit.start():hit.end()],
            "extracted_title": title, "extracted_price": price, "original_assistant_content": content,
            "augmented_assistant_content": f"{content} [Item {asin_raw.upper()}: {title} | {price}]"}


def encode_current_prefix(tokenizer, messages, max_length):
    """线上/离线共用：只编码当前动作以前的历史，返回实际输入与溯源。"""
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("评分前缀必须结束于当前 assistant 动作")
    if not messages[-1]["content"].strip():
        return None, {"skip_reason": "empty_response", "asin_length_fallback": False}
    history = [dict(m) for m in messages]
    for message in history[:-1]:
        if message["role"] == "assistant" and not message["content"].strip():
            message["content"] = "(Empty assistant reply)"
    dropped = 0
    while True:
        try:
            raw = tokenize_trajectory(tokenizer, history, max_length)
        except TrajectoryTooLong:
            if len(history) <= 3:
                return None, {"skip_reason": "prefix_too_long", "asin_length_fallback": False,
                              "dropped_exchanges": dropped}
            del history[1:3]
            dropped += 1
            continue
        augmented = [dict(m) for m in history]
        provenance = []
        for index, message in enumerate(history):
            if message["role"] == "assistant":
                record = extract_preclick_asin_snippet(history, index)
                if record:
                    augmented[index]["content"] = record["augmented_assistant_content"]
                    provenance.append(record)
        encoded, fallback = raw, False
        if provenance:
            try:
                encoded = tokenize_trajectory(tokenizer, augmented, max_length)
            except TrajectoryTooLong:
                fallback = True
        trace = {"skip_reason": None, "asin_length_fallback": fallback,
                 "dropped_exchanges": dropped, "input_ids_sha256": token_sha(encoded["input_ids"]),
                 "score_token_index": encoded["turn_end_indices"][-1],
                 "current_assistant_message_index": len(history) - 1,
                 "position_convention": "current_prefix_local", "asin_context_provenance": provenance}
        return encoded, trace


def prepare_prefixes(tokenizer, messages, max_length):
    """后续消息不参与任何较早动作的编码或接纳决定。"""
    if not isinstance(messages, list) or not messages or messages[0].get("role") != "system":
        raise ValueError("样本须包含 system 与完整对话历史")
    if len(messages) % 2 != 1:
        raise ValueError("完整样本须结束于 assistant")
    for index, message in enumerate(messages):
        role = "system" if index == 0 else ("user" if index % 2 else "assistant")
        if message.get("role") != role or not isinstance(message.get("content"), str):
            raise ValueError("对话角色或内容无效")
    prefixes = [encode_current_prefix(tokenizer, messages[:index + 1], max_length)
                for index in range(2, len(messages), 2)]
    if not prefixes:
        raise ValueError("样本缺少 assistant 动作")
    return prefixes


def rollout_messages(row):
    """推理仅还原 observation/response，不读取标签或依赖旧 runtime 缺少的 data。"""
    from webshop_lab.model import SYSTEM
    if not isinstance(row.get("instruction"), str) or not row["instruction"].strip():
        raise ValueError("原始轨迹缺少购物指令")
    if not isinstance(row.get("steps"), list) or not row["steps"]:
        raise ValueError("原始轨迹缺少步骤")
    messages = [{"role": "system", "content": SYSTEM + "\nShopping instruction: " + row["instruction"].strip()}]
    for step in row["steps"]:
        if not all(isinstance(step.get(key), str) for key in ("observation", "response")):
            raise ValueError("原始轨迹须包含 observation/response 文本")
        messages.extend([{"role": "user", "content": step["observation"]},
                         {"role": "assistant", "content": step["response"]}])
    return messages


def validate_pv_predictions(rows):
    """局部评分位裁剪后允许变小，禁止伪装成完整轨迹的全局 token 下标。"""
    seen = set()
    if not isinstance(rows, list) or not rows:
        raise ValueError("预测不能为空")
    for row in rows:
        task = row.get("task_id")
        scores, values, traces = row.get("step_scores"), row.get("v_prefix"), row.get("prefix_traces")
        if type(task) is not int or task in seen:
            raise ValueError("任务 ID 无效或重复")
        seen.add(task)
        if (not isinstance(scores, list) or not scores or not isinstance(values, list)
                or not isinstance(traces, list) or len(scores) != len(values) or len(scores) != len(traces)):
            raise ValueError("前缀预测、奖励与溯源数量不一致")
        if (row.get("position_convention") != "per_prefix_local"
                or row.get("turn_end_indices") != [trace.get("score_token_index") for trace in traces]):
            raise ValueError("评分位置必须对应每个实际前缀，不能伪装成全局位置")
        previous = 0.0
        for value, reward, trace in zip(values, scores, traces):
            if not all(type(x) in (int, float) and math.isfinite(x) for x in (value, reward)):
                raise ValueError("预测值必须有限")
            if not math.isclose(value - previous, reward, abs_tol=1e-7, rel_tol=1e-7):
                raise ValueError("奖励不是相邻前缀价值差")
            if trace.get("skip_reason") is None:
                if (not re.fullmatch(r"[0-9a-f]{64}", trace.get("input_ids_sha256", ""))
                        or type(trace.get("score_token_index")) is not int or trace["score_token_index"] < 0):
                    raise ValueError("缺少实际评分前缀身份")
            elif trace["skip_reason"] not in ("empty_response", "prefix_too_long") or value != previous:
                raise ValueError("跳过评分时必须保留上一价值")
            previous = value
        if not math.isclose(row["total"], previous, rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError("总分须为最后前缀价值")
    return rows


def compare_pv_predictions(actual, expected, atol, rtol):
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in (atol, rtol)):
        raise ValueError("容差须为有限非负数")
    validate_pv_predictions(actual)
    validate_pv_predictions(expected)
    if len(actual) != len(expected):
        raise ValueError("重载样本数不一致")
    for left, right in zip(actual, expected):
        if left["task_id"] != right["task_id"] or left["prefix_traces"] != right["prefix_traces"]:
            raise ValueError("重载任务或逐步输入身份不一致")
        if any(not math.isclose(a, b, abs_tol=atol, rel_tol=rtol)
               for a, b in zip(left["v_prefix"] + left["step_scores"] + [left["total"]],
                               right["v_prefix"] + right["step_scores"] + [right["total"]])):
            raise ValueError("重载数值超出容差")
