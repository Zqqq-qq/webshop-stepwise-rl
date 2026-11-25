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


if __name__=='__main__': unittest.main()
