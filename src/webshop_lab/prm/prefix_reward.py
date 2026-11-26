"""通过外置因果编码器调用冻结的旧 runtime，不修改历史检查点。"""
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

_helper = Path(__file__).resolve().parents[3] / "scripts" / "prm_prefix_value_support.py"
_spec = importlib.util.spec_from_file_location("prm_prefix_value_support", _helper)
SUPPORT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SUPPORT)


def encode_current_prefix(tokenizer, messages, max_length):
    """保留原在线辅助接口；完整溯源由同一 SUPPORT 编码器提供。"""
    encoded, trace = SUPPORT.encode_current_prefix(tokenizer, messages, max_length)
    return encoded, trace["asin_length_fallback"]


class OnlinePrefixReward:
    def __init__(self, config, checkpoint, runtime, contract, contract_sha256):
        SUPPORT.verify_contract(checkpoint, contract, contract_sha256)
        runtime = Path(runtime).resolve()
        self.last_trace = None
        self.worker = subprocess.Popen(
            [sys.executable, "-c", 'import runpy, sys; sys.argv.pop(0); runpy.run_path(sys.argv[0], run_name="__main__")',
             str(Path(__file__).resolve()), str(Path(config).resolve()), str(Path(checkpoint).resolve()),
             str(Path(contract).resolve()), contract_sha256],
            cwd=runtime, env={**os.environ, "PYTHONPATH": str(runtime / "src")},
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")
        try:
            if self._receive() != {"ready": True}:
                raise RuntimeError("评分器子进程握手失败")
        except BaseException:
            self.close()
            raise

    def _receive(self):
        line = self.worker.stdout.readline()
        if not line:
            raise RuntimeError("评分器子进程停止，请检查 traceback")
        return json.loads(line)

    def __call__(self, messages):
        self.worker.stdin.write(json.dumps(messages, ensure_ascii=False) + "\n")
        self.worker.stdin.flush()
        result = self._receive()
        self.last_trace = result
        return result["value"], result["asin_length_fallback"]

    def close(self):
        self.worker.stdin.close()
        try:
            self.worker.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.worker.kill()
            self.worker.wait()
        finally:
            self.worker.stdout.close()


def serve(config, checkpoint, contract_path, contract_sha256):
    import torch
    from webshop_lab.prm.config import load_config
    from webshop_lab.prm.runtime import build_prm, tensor_batch

    contract = SUPPORT.verify_contract(checkpoint, contract_path, contract_sha256)
    cfg = load_config(config)
    if cfg.max_length != contract["max_length"]:
        raise ValueError("评分配置窗口与契约不一致")
    model, tokenizer, _, _ = build_prm(cfg, checkpoint, "cuda", False)
    device = next(model.parameters()).device
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        prefix = json.loads(line)
        encoded, trace = SUPPORT.encode_current_prefix(tokenizer, prefix, cfg.max_length)
        value = None
        if encoded is not None:
            batch = tensor_batch([encoded], tokenizer.pad_token_id, device, False)
            with torch.no_grad():
                value = float(model(**batch)["step_scores"][0, -1])
            if not math.isfinite(value):
                raise ValueError("评分器返回非有限前缀价值")
        print(json.dumps({**trace, "value": value,
                          "scoring_contract_sha256": contract_sha256}), flush=True)


if __name__ == "__main__":
    serve(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
