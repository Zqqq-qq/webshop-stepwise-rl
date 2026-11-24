"""CPU tests for the memory-test acceptance calculation."""
import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).resolve().parents[1]/'scripts'/'run_prm_memory_probe.py'
spec=importlib.util.spec_from_file_location('memory_probe',path)


class MemoryMetricsTests(unittest.TestCase):
    def metrics(self,predictions,labels):
        self.assertTrue(path.exists(),'Memory probe implementation is missing')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module.memory_metrics(predictions,labels)

    def test_exact_fit_passes(self):
        result=self.metrics([{'task_id':1,'total':0.0},{'task_id':2,'total':1.0}],{1:0.,2:1.})
        self.assertTrue(result['passed'])
        self.assertEqual(result['mse'],0.)

    def test_low_mean_error_does_not_hide_one_bad_sample(self):
        pred=[{'task_id':i,'total':0.11 if i==0 else 0.} for i in range(16)]
        result=self.metrics(pred,{i:0. for i in range(16)})
        self.assertLess(result['mse'],0.001)
        self.assertFalse(result['passed'])

    def test_missing_or_duplicate_prediction_is_rejected(self):
        for pred in [[{'task_id':1,'total':0.}],
                     [{'task_id':1,'total':0.},{'task_id':1,'total':0.}]]:
            with self.assertRaises(ValueError):self.metrics(pred,{1:0.,2:1.})

    def test_nonfinite_prediction_is_rejected(self):
        with self.assertRaises(ValueError):self.metrics([{'task_id':1,'total':float('nan')}],{1:0.})


if __name__=='__main__':unittest.main()
