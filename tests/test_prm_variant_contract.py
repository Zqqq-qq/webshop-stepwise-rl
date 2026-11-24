"""方案身份测试，无Torch；并未给现行runtime增加hidden开关。"""
import dataclasses
import json
from pathlib import Path
import unittest
from webshop_lab.prm.variant_contract import RepresentationContract,check_manifest_header,LEGACY_SCORE
from webshop_lab.prm.config import PRMConfig


class VariantContractTests(unittest.TestCase):
    def setUp(self):self.hidden=RepresentationContract('final_hidden_state',1536);self.identity={'fixed':'cpu-fixture'}

    def test_hidden_uses_hidden_dimension_not_vocabulary(self):
        self.hidden.validate_dimensions(151936,1536)
        with self.assertRaises(ValueError):RepresentationContract('final_hidden_state',151936).validate_dimensions(151936,1536)

    def test_both_proposal_envelopes_have_valid_existing_runtime_config(self):
        root=Path(__file__).resolve().parents[1]/'configs/prm/proposals'
        for name in ('logits-linear','hidden-linear'):
            plan=json.loads((root/(name+'-v1.json')).read_text())
            PRMConfig.from_dict(plan['runtime_config'])
            RepresentationContract(**plan['architecture']).validate_dimensions(151936,1536)
            self.assertFalse(plan['execution_authorized'])
            self.assertIsNone(plan['data_lock']['samples_sha256'])
            # 外层proposal禁止当作旧CLI配置静默执行。
            with self.assertRaises(ValueError):PRMConfig.from_dict(plan)

    def test_v2_header_roundtrip_and_wrong_head_rejection(self):
        header=self.hidden.manifest_header(self.identity,151936,1536)
        self.assertEqual(check_manifest_header(header,self.hidden,self.identity,151936,1536),'v2_header_only_not_full_reload')
        for key,value in [('score_definition',LEGACY_SCORE),('schema_version',3),('hidden_size',2048),('identity',{})]:
            wrong=dict(header,**{key:value})
            with self.assertRaises(ValueError):check_manifest_header(wrong,self.hidden,self.identity,151936,1536)
        with self.assertRaises(ValueError):check_manifest_header(header,RepresentationContract('vocab_logits',151936),self.identity,151936,1536)

    def test_legacy_manifest_not_rewritten_and_never_loaded_as_hidden(self):
        legacy={'schema_version':1,'identity':self.identity,'vocab_size':151936,'score_definition':LEGACY_SCORE}
        snapshot=json.dumps(legacy,sort_keys=True)
        self.assertEqual(check_manifest_header(legacy,RepresentationContract('vocab_logits',151936),self.identity,151936,1536),'legacy_v1_requires_pinned_runtime')
        with self.assertRaises(ValueError):check_manifest_header(legacy,self.hidden,self.identity,151936,1536)
        self.assertEqual(json.dumps(legacy,sort_keys=True),snapshot)

    def test_simultaneous_definition_changes_rejected(self):
        for kwargs in ({'head_type':'mlp'},{'reduction':'min'},{'score_position':'content_last_token'},{'dtype':'bfloat16'}):
            with self.assertRaises(ValueError):dataclasses.replace(self.hidden,**kwargs)
