"""实际共用比较器拒绝伪重载；这些CPU测试不表示GPU模型通过。"""
import copy
import unittest
from webshop_lab.prm.predictions import compare_predictions


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.rows=[{'task_id':1,'turn_end_indices':[3,8],'step_scores':[.2,.3],'total':.5},
                   {'task_id':2,'turn_end_indices':[4],'step_scores':[.1],'total':.1}]

    def test_finite_aligned_scores_and_total_match(self):
        compare_predictions(self.rows,self.rows,1e-5,1e-5)

    def test_nonfinite_steps_or_total_on_either_side_rejected(self):
        for value in (float('nan'),float('inf'),-float('inf')):
            for key in ('step_scores','total'):
                bad=copy.deepcopy(self.rows)
                if key=='step_scores':bad[0][key][0]=value
                else:bad[0][key]=value
                for a,b in ((bad,self.rows),(self.rows,bad)):
                    with self.assertRaises(ValueError):compare_predictions(a,b,1e-5,1e-5)

    def test_empty_missing_extra_duplicate_tasks_rejected(self):
        variants=[[],self.rows[:1],self.rows+[dict(self.rows[0],task_id=3)],
                  [self.rows[0],self.rows[0]]]
        for bad in variants:
            with self.assertRaises(ValueError):compare_predictions(bad,self.rows,1e-5,1e-5)

    def test_alignment_short_long_and_total_mismatch_rejected(self):
        for key,value in (('step_scores',[.2]),('step_scores',[.2,.3,.4]),
                          ('turn_end_indices',[3,9]),('total',.7)):
            bad=copy.deepcopy(self.rows);bad[0][key]=value
            with self.assertRaises(ValueError):compare_predictions(bad,self.rows,1e-5,1e-5)

    def test_invalid_tolerance_rejected(self):
        for v in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):compare_predictions(self.rows,self.rows,v,1e-5)
