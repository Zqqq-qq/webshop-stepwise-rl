"""阻止已登记 PV 载荷被当前版本旧 sum 推理入口静默解释。"""
import hashlib
import json
from pathlib import Path

# 原归档保持不变；这些 manifest 虽写 sum，实际来自已公布的 PV 训练。
REGISTERED_PV_MANIFESTS = {
    '6dd7110532a808856a7612213ee085e1f12248c12ae0650495dac44aaaf96750',
    '826f9cce3a51365ec6bf02fb10574ca4990010a8927e9082b5a0383fd028b01c',
    '8ef74fd81ee62d3eaeee708852133d0ff2703153151449054df82022ab58d6d4',
    '95feaf82b667cf22ec2cce39ded51f403832b671fdc3dab233f2b692043f9006',
    'b3906493cd162924cc556d653dad8f903b3219a23bbf7d23aec3afe527e684c9',
}


REGISTERED_PV_HEADS = {
    '0bcfdaa47afaf31f76fa79ea8bdce415cd8ec551f81f34744ffc6a2db9be62a6',
    '54fc69d5667d3222b5bbb938a78f536abb0a1d36011ac1999e559d0ddd799a17',
    '6a5d39a8ab6b8ced55e5156ff8b8bacc505a8b3141b586c5e5cfb5cc6d64ad1f',
    'a234fd6d1a4d89a2214064f2e17ac7f85896126f7fa018df18eec8b5afe2a23e',
    'bce9c201da303beed53d9f1621c44a6370418da40b5647e62ffd37dc2dc0131a',
}


def reject_registered_pv_checkpoint(checkpoint):
    raw = (Path(checkpoint) / "prm-manifest.json").read_bytes()
    manifest = json.loads(raw)
    if (hashlib.sha256(raw).hexdigest() in REGISTERED_PV_MANIFESTS
            or manifest.get("files_sha256", {}).get("head.pt") in REGISTERED_PV_HEADS
            or "prefix_value" in manifest.get("score_definition", "")):
        raise ValueError("前缀价值检查点不能用旧 sum 入口；请使用带冻结评分契约的 score_prefix_value_prm.py")
