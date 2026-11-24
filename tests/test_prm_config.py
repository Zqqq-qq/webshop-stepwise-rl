import hashlib
import tempfile
import unittest
from pathlib import Path

from webshop_lab.prm.config import PRMConfig, verify_input_files


def example(**kwargs):
    data = dict(model='input/base', revision='a'*40, base_weights_sha256='b'*64,
                tokenizer='input/tokenizer', tokenizer_revision='a'*40,
                tokenizer_files_sha256={'tokenizer.json': 'c'*64}, output_dir='output/prm')
    data.update(kwargs)
    return data


class PRMConfigTests(unittest.TestCase):
    def test_roundtrip_and_stable_fingerprint(self):
        cfg = PRMConfig.from_dict(example())
        self.assertEqual(cfg, PRMConfig.from_dict(cfg.to_dict()))
        self.assertEqual(cfg.fingerprint(), PRMConfig.from_dict(dict(reversed(list(cfg.to_dict().items())))).fingerprint())

    def test_invalid_config_rejected(self):
        for change in [dict(revision='main'), dict(base_weights_sha256='bad'), dict(max_length=True),
                       dict(lora_r=0), dict(policy_dtype='auto'), dict(load_in_4bit='false'),
                       dict(load_in_4bit=True), dict(target_modules=['q_proj', 'q_proj']),
                       dict(score_position='last_nonpad'), dict(unknown=True),
                       dict(tokenizer_files_sha256={'../secret': 'a'*64}),
                       dict(output_dir='input/base/nested')]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                PRMConfig.from_dict(example(**change))

    def test_b1_identity_is_required_and_never_implicit(self):
        with self.assertRaises(ValueError):
            PRMConfig.from_dict(example(initialization='sft_b1'))
        with self.assertRaises(ValueError):
            PRMConfig.from_dict(example(sft_adapter='input/b1'))
        cfg = PRMConfig.from_dict(example(initialization='sft_b1', sft_adapter='input/b1',
            sft_adapter_files_sha256={'adapter_model.safetensors': 'a'*64, 'adapter_config.json': 'b'*64}))
        self.assertEqual(cfg.initialization, 'sft_b1')
        with self.assertRaises(ValueError):
            PRMConfig.from_dict({**cfg.to_dict(), 'output_dir': 'input/b1'})

    def test_verify_payload_then_detect_changed_tokenizer(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d);base=p/'base';tok=p/'tok';base.mkdir();tok.mkdir()
            (base/'model.safetensors').write_bytes(b'weights')
            (tok/'tokenizer.json').write_bytes(b'tokenizer')
            digest=lambda b:hashlib.sha256(b).hexdigest()
            cfg=PRMConfig.from_dict(example(model=str(base),tokenizer=str(tok),output_dir=str(p/'out'),
                base_weights_sha256=digest(b'weights'),tokenizer_files_sha256={'tokenizer.json':digest(b'tokenizer')}))
            self.assertEqual(len(verify_input_files(cfg)),2)
            (tok/'tokenizer.json').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                verify_input_files(cfg)
