"""Regression checks for saved-result statistics and split-run records."""
import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'


def module(name):
    spec=importlib.util.spec_from_file_location(name,SCRIPTS/(name+'.py'))
    result=importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(SCRIPTS))
    try: spec.loader.exec_module(result)
    finally: sys.path.pop(0)
    return result


try:
    import numpy as np
except ImportError:
    np=None


@unittest.skipIf(np is None,'NumPy required for numerical evaluation tests')
class EvaluationStatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.m=module('audit_prm_historical_dev')

    def test_rank_ties_use_midrank_and_constants_have_no_correlation(self):
        np.testing.assert_array_equal(self.m.ranks(np.array([5.,1.,5.,3.])),[3.5,1,3.5,2])
        result=self.m.metrics(np.ones(4),np.array([0.,1.,0.,1.]))
        self.assertIsNone(result['pearson_r']);self.assertIsNone(result['spearman_rho'])

    def test_baseline_fit_matches_known_nonconstant_data(self):
        mean,groups,intercept,slope=self.m.fit_baselines(np.array([1.,1.,3.,3.]),np.array([0.,0.,1.,1.]))
        self.assertEqual(mean,.5);self.assertEqual(groups,{1:0.,3:1.})
        self.assertEqual(intercept,-.5);self.assertEqual(slope,.5)
        self.assertEqual(groups.get(7,mean),.5)

    def test_paired_bootstrap_preserves_identical_predictions_and_sign(self):
        y=np.array([0.,1.,0.,1.]); p=np.array([.1,.7,.3,.5])
        self.assertEqual(self.m.bootstrap_delta(p,p,y,repeats=100),[0.,0.,0.])
        worse=self.m.bootstrap_delta(y+1,y,y,repeats=100)
        self.assertEqual(worse,[1.,1.,1.])

    def test_observed_mse_delta_is_not_bootstrap_monte_carlo_mean(self):
        y=np.zeros(4); p=np.array([0.,0.,0.,4.])
        result,draw_mean=self.m.bootstrap_delta(p,y,y,repeats=4,return_draw_mean=True)
        self.assertEqual(result[0],4.)
        self.assertNotEqual(result[0],draw_mean)

    def test_historical_models_align_by_task_instead_of_dictionary_order(self):
        rows=[dict(task_id=1,terminal_reward=0.,termination='purchase'),
              dict(task_id=2,terminal_reward=1.,termination='max_steps')]
        old={2:dict(total=0.,turn_end_indices=[5]),1:dict(total=1.,turn_end_indices=[3])}
        csv={('dev',r['task_id']):dict(label=str(r['terminal_reward']),termination=r['termination']) for r in rows}
        result=self.m.compare_historical_models(np.array([0.,1.]),old,rows,{1:[3],2:[5]},csv)
        self.assertEqual(result['observed_delta_and_ci95'],[-1.,-1.,-1.])
        csv['dev',1]['termination']='max_steps'
        with self.assertRaises(ValueError):
            self.m.compare_historical_models(np.array([0.,1.]),old,rows,{1:[3],2:[5]},csv)

    def test_historical_model_comparison_rejects_missing_tasks(self):
        with self.assertRaises(ValueError):
            self.m.compare_historical_models(np.array([0.]),{},
                [dict(task_id=1,terminal_reward=0.,termination='purchase')],{1:[3]}, {})

    def test_source_bundle_checksum_is_checked_before_clone(self):
        with self.assertRaisesRegex(ValueError,'bundle SHA'):
            self.m.verify_source_bundle(b'not-the-original-bundle','unused-checkout',{})


class SplitRecordTests(unittest.TestCase):
    def setUp(self):
        self.m=module('audit_prm_split_saved')
        self.labels={i:float(i%2) for i in range(8)}
        self.summary={'optimizer_steps':120,'samples_consumed':960,
            'optimization':{'gradient_accumulation_steps':8,'lr_scheduler':'constant',
                            'learning_rate':1e-5,'head_learning_rate':1e-5},'records':[]}
        for i in range(120):
            self.summary['records'].append(dict(optimizer_step=i+1,task_ids=list(self.labels),
                terminal_rewards=list(self.labels.values()),samples_consumed=(i+1)*8,
                totals_before_update=[0.]*8,loss=.5,gradient_norm=2.,gradient_clipped=True,
                learning_rate=1e-5,head_learning_rate=1e-5))

    def test_clip_flags_are_checked_against_real_norms(self):
        self.assertEqual(self.m.check_records(self.summary,self.labels)['clipped_updates'],120)
        self.summary['records'][2]['gradient_clipped']=False
        with self.assertRaises(ValueError): self.m.check_records(self.summary,self.labels)

    def test_loss_and_task_order_cannot_silently_change(self):
        self.summary['records'][2]['loss']=.49
        with self.assertRaises(ValueError): self.m.check_records(self.summary,self.labels)
        self.summary['records'][2]['loss']=.5
        self.summary['records'][2]['task_ids'].reverse()
        with self.assertRaises(ValueError): self.m.check_records(self.summary,self.labels)

    def test_wrong_actual_head_learning_rate_is_rejected(self):
        self.summary['records'][2]['head_learning_rate']=1e-4
        with self.assertRaises(ValueError): self.m.check_records(self.summary,self.labels)


class HoldoutIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.m=module('audit_prm_holdout_protocol')

    def test_instruction_prefix_case_and_whitespace_are_same_identity(self):
        self.assertEqual(self.m.instruction_hash('  Instruction:  RED   Mug '),
                         self.m.instruction_hash('red mug'))

    def test_reserved_and_seen_instructions_and_duplicate_candidates_are_excluded(self):
        goals={1:'old',2:'Instruction: OLD',3:'reserved',4:'new',5:'NEW',6:'unfinished',7:'reserved'}
        candidate,overlap,duplicate,eligible,selected=self.m.isolate_candidates(
            [6,5,4,3,2,1],[1],[7],goals,6)
        self.assertEqual(candidate,[2,3,4,5,6]);self.assertEqual(overlap,[2,3])
        self.assertEqual(duplicate,[5]);self.assertEqual(eligible,[4]);self.assertEqual(selected,[4])

    def test_candidate_ranking_is_independent_of_input_iteration_order(self):
        goals={t:str(t) for t in range(200)}
        a=self.m.isolate_candidates(range(200),[],[],goals,None)
        b=self.m.isolate_candidates(reversed(range(200)),[],[],goals,None)
        self.assertEqual(a,b);self.assertEqual(len(a[-1]),160)

    def test_replacement_preserves_retained_order_and_excludes_external_instruction_twins(self):
        goals={t:'task '+str(t) for t in range(10)}
        goals[99]='Instruction: TASK 1'
        goals[100]='task 2'
        eligible=list(range(10))
        orig=sorted(eligible,key=lambda t:(self.m.sha(('42:'+str(t)).encode('ascii')),t))[:5]
        exposed_elig=[orig[1]]
        r_id=self.m.select_replacement_candidates(eligible,orig,exposed_elig,goals,count=5)
        self.assertEqual(r_id['retained_count'],4)
        self.assertEqual(r_id['removed_from_original_160'],[orig[1]])
        self.assertNotIn(orig[1],r_id['selected_task_ids'])
        self.assertEqual(r_id['selected_ids_sha256'],
                         self.m.sha(self.m.canonical_ids_bytes(r_id['selected_task_ids'])))
        if 1 in r_id['cross_goal_instruction_twins_in_remaining'] or str(1) in r_id['cross_goal_instruction_twins_in_remaining']:
            self.assertEqual(r_id['cross_goal_instruction_twins_in_remaining']['1'],[99])
        r_twin=self.m.select_replacement_candidates(
            eligible,orig,exposed_elig,goals,all_exposed_ids=set(exposed_elig)|{99,100},count=5)
        self.assertNotIn(1,r_twin['selected_task_ids'])
        self.assertNotIn(2,r_twin['selected_task_ids'])
        for t in (1,2):
            if t not in exposed_elig:
                self.assertIn(t,r_twin['excluded_by_instruction_twin'])

    def test_replacement_rejects_duplicate_or_unknown_exposed_ids(self):
        goals={t:str(t) for t in range(10)}
        with self.assertRaisesRegex(ValueError,'duplicates'):
            self.m.select_replacement_candidates(range(10),[0,1],[2,2],goals,count=2)
        with self.assertRaisesRegex(ValueError,'outside eligible'):
            self.m.select_replacement_candidates(range(10),[0,1],[99],goals,count=2)

    def test_device_handoff_verifier_checks_shas_skipped_and_twin_isolation(self):
        import json, tempfile
        goals={t:'goal '+str(t) for t in range(8)}
        eligible=list(range(8))
        orig=sorted(eligible,key=lambda t:(self.m.sha(('42:'+str(t)).encode('ascii')),t))[:4]
        target_in_orig=orig[2]
        exposed_orig=orig[0]
        goals[77]='  Instruction: GOAL '+str(target_in_orig)+'  '
        clean=self.m.select_replacement_candidates(
            eligible,orig,[exposed_orig],goals,all_exposed_ids={exposed_orig,77},count=4)
        self.assertEqual(clean['excluded_by_instruction_twin'],[target_in_orig])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            files={
                'preflight-summary.json':json.dumps({'skipped':0}).encode(),
                'full-history-audit.json':json.dumps({'skipped':0,'records':[{'task_id':exposed_orig},{'task_id':77}]}).encode(),
                'eligible-history-audit.json':json.dumps({'skipped':0,'records':[{'task_id':exposed_orig}]}).encode(),
                'holdout-replacement-candidate.json':self.m.canonical_ids_bytes(clean['selected_task_ids']),
            }
            shas={}
            for name,raw in files.items():
                (root/name).write_bytes(raw);shas[name]=self.m.sha(raw)
            verified=self.m.verify_device_handoff(root,shas,eligible,orig,goals,count=4)
            self.assertTrue(verified['device_handoff_received'])
            self.assertEqual(verified['excluded_by_instruction_twin'],[target_in_orig])
            # Reject SHA tampering
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):
                self.m.verify_device_handoff(root,dict(shas,**{'preflight-summary.json':'0'*64}),eligible,orig,goals,count=4)
            # Reject non-zero skipped count
            bad_pre=json.dumps({'skipped':1}).encode()
            (root/'preflight-summary.json').write_bytes(bad_pre)
            bad_shas=dict(shas,**{'preflight-summary.json':self.m.sha(bad_pre)})
            with self.assertRaisesRegex(ValueError,'skipped'):
                self.m.verify_device_handoff(root,bad_shas,eligible,orig,goals,count=4)
            # Restore preflight, unconditionally reject stale candidate that forgot external twin 77 -> target_in_orig
            (root/'preflight-summary.json').write_bytes(files['preflight-summary.json'])
            stale=self.m.select_replacement_candidates(eligible,orig,[exposed_orig],goals,count=4)
            self.assertIn(target_in_orig,stale['selected_task_ids'])
            stale_raw=self.m.canonical_ids_bytes(stale['selected_task_ids'])
            (root/'holdout-replacement-candidate.json').write_bytes(stale_raw)
            stale_shas=dict(shas,**{'holdout-replacement-candidate.json':self.m.sha(stale_raw)})
            with self.assertRaisesRegex(ValueError,'disagree'):
                self.m.verify_device_handoff(root,stale_shas,eligible,orig,goals,count=4)

    def test_extract_actual_task_ids_ignores_plan_only_selected_ids(self):
        payload={'selected_ids':[1,2,3],'records':[{'task_id':4},{'sample_keys':[[5,0]]}]}
        self.assertEqual(self.m.extract_actual_task_ids(payload),{4,5})


if __name__=='__main__': unittest.main()
