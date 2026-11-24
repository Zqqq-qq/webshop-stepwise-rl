"""Adversarial saved-record checks. These fixtures are synthetic, not GPU runs."""
import copy
import importlib.util
from pathlib import Path
import unittest
import tempfile
import zipfile

spec=importlib.util.spec_from_file_location('hidden_audit',Path(__file__).resolve().parents[1]/
                                         'scripts/audit_prm_hidden_results.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class HiddenAuditTests(unittest.TestCase):
    def setUp(self):
        ids=[351,95,5478,3542,359,5031,0,6225]
        self.labels=dict(zip(ids,[2/3,0,0,0,2/3,0,1,1]))
        self.positions={t:[3] for t in ids}
        self.ref=[{'task_id':t,'turn_end_indices':[3],'step_scores':[.2],'total':.2} for t in ids]
        self.summary={'identity':{'source_commit':'synthetic'},'samples_sha256':m.SAMPLES_SHA,
                      'train_ids_sha256':m.TRAIN_SHA,'total_samples':8,'retained_samples':8,
                      'rejected':[],'optimizer_steps':2,'samples_consumed':16,
                      'optimization':{'gradient_accumulation_steps':8,'weight_decay':.01,
                                      'max_grad_norm':1.0},
                      'peak_allocated_gib':0,'peak_reserved_gib':0,'records':[]}
        for step in (1,2):
            self.summary['records'].append({'optimizer_step':step,'samples_consumed':step*8,
                'task_ids':ids,'terminal_rewards':list(self.labels.values()),'gradient_norm':2.,
                'loss':m.score(dict.fromkeys(ids,0.),self.labels)['mse'],
                'totals_before_update':[0.]*8,'gradient_clipped':True})
        self.reload={'predictions':copy.deepcopy(self.ref),'identity':{'source_commit':'synthetic'},
                     'reference_match':True,'rejected':[],'total_samples':8}

    def audit(self):
        return m.audit_run(self.summary,self.ref,self.reload,self.labels,self.positions)

    def test_complete_records_recompute_metrics_and_groups(self):
        result=self.audit()
        self.assertAlmostEqual(result['mse'],m.score(dict.fromkeys(self.labels,.2),self.labels)['mse'])
        self.assertEqual(result['segment_clip_ratio'],1)
        self.assertEqual(result['full_minus_zero_group_mean'],0)
        self.assertFalse(result['memory_gate_passed'])

    def test_saved_reference_and_reload_must_both_cover_every_task(self):
        self.reload['predictions'].pop()
        with self.assertRaises(ValueError): self.audit()

    def test_duplicate_task_cannot_replace_missing_task(self):
        self.ref[-1]=self.ref[0]
        with self.assertRaises(ValueError): self.audit()

    def test_nonfinite_reload_is_rejected_even_with_true_match_flag(self):
        self.reload['predictions'][0]['step_scores']=[float('nan')]
        with self.assertRaises(ValueError): self.audit()

    def test_wrong_score_position_is_rejected(self):
        self.ref[0]['turn_end_indices']=[2]
        with self.assertRaises(ValueError): self.audit()

    def test_wrong_clipping_flag_is_rejected(self):
        self.summary['records'][0]['gradient_clipped']=False
        with self.assertRaises(ValueError): self.audit()

    def test_changed_training_labels_are_rejected(self):
        self.summary['records'][0]['terminal_rewards']=[0]*8
        with self.assertRaises(ValueError): self.audit()

    def test_wrong_exposure_counter_is_rejected(self):
        self.summary['records'][0]['samples_consumed']=16
        with self.assertRaises(ValueError): self.audit()

    def test_missing_optimizer_step_is_rejected(self):
        self.summary['records'][1]['optimizer_step']=3
        with self.assertRaises(ValueError): self.audit()

    def test_recorded_loss_is_independently_checked(self):
        self.summary['records'][0]['loss']=42.
        with self.assertRaises(ValueError): self.audit()

    def test_resume_segment_does_not_claim_cumulative_clipping_coverage(self):
        for row,step in zip(self.summary['records'],(17,18)):
            row['optimizer_step']=step;row['samples_consumed']=step*8
        self.summary['optimizer_steps']=18;self.summary['samples_consumed']=144
        result=self.audit()
        self.assertEqual(result['segment_optimizer_steps'],2)
        self.assertEqual(result['cumulative_optimizer_steps'],18)
        self.assertEqual(result['segment_clip_count'],2)

    def test_continuation_requires_exact_predecessor_parameter_digests(self):
        previous = copy.deepcopy(self.summary)
        previous['optimizer_steps'] = 0
        previous['parameters_after'] = {'head':'saved-head', 'lora':'saved-lora'}
        self.summary['parameters_before'] = {'head':'reset-head', 'lora':'saved-lora'}
        with self.assertRaises(ValueError): m.validate_continuation(self.summary, previous, 2)
        self.summary['parameters_before'] = previous['parameters_after'].copy()
        m.validate_continuation(self.summary, previous, 2)

    def test_continuation_cannot_skip_first_update_or_change_settings(self):
        previous = copy.deepcopy(self.summary)
        previous['optimizer_steps'] = 0
        previous['parameters_after'] = {'head':'h', 'lora':'l'}
        self.summary['parameters_before'] = previous['parameters_after'].copy()
        self.summary['records'].pop(0)
        with self.assertRaises(ValueError): m.validate_continuation(self.summary, previous, 2)
        self.summary['records'].insert(0, {'optimizer_step':1})
        self.summary['optimization']['learning_rate'] = .1
        with self.assertRaises(ValueError): m.validate_continuation(self.summary, previous, 2)

    def test_group_components_sum_to_original_mse(self):
        positions={t:[1]*(3 if t not in (5478,3542) else 10) for t in self.labels}
        values={t:i/10 for i,t in enumerate(self.labels)}
        parts=m.group_error_components(values,self.labels,positions)
        self.assertAlmostEqual(sum(parts.values()),m.score(values,self.labels)['mse'])


class ArchiveRootTests(unittest.TestCase):
    def check_archive(self, entries):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'evidence.zip'
            with zipfile.ZipFile(path,'w') as z:
                for name,data in entries.items(): z.writestr(name,data)
            return m.verified_archive(path,m.sha(path.read_bytes()))

    def test_nested_root_and_dot_slash_checksums(self):
        files,checks=self.check_archive({'artifacts/run/SHA256SUMS':m.sha(b'data')+'  ./sample.json\n',
                                        'artifacts/run/sample.json':b'data'})
        self.assertEqual(files['sample.json'],b'data')
        self.assertEqual(len(checks),1)

    def test_payload_outside_checksum_root_is_rejected(self):
        with self.assertRaises(ValueError):
            self.check_archive({'artifacts/run/SHA256SUMS':'','other/payload':b'x'})

    def test_duplicate_normalized_checksum_entries_are_rejected(self):
        line=m.sha(b'data')+'  '
        with self.assertRaises(ValueError):
            self.check_archive({'run/SHA256SUMS':line+'./sample\n'+line+'sample\n',
                                'run/sample':b'data'})


if __name__=='__main__': unittest.main()
