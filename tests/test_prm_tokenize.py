"""Real pinned tokenizer checks; optional and offline, never download GPU weights."""
import copy
import unittest

from webshop_lab.prm.tokenize import TrajectoryTooLong, collate_trajectories, tokenize_trajectory

REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'


class QwenPRMTokenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from transformers import AutoTokenizer
            cls.tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',
                revision=REVISION, local_files_only=True, trust_remote_code=False)
        except (ImportError, OSError) as exc:
            raise unittest.SkipTest('Install Transformers and cache the pinned Qwen tokenizer to run real template tests') from exc

    def messages(self):
        return [{'role':'system','content':'Shopping instruction: 黑色鞋，预算30元'},
                {'role':'user','content':'Search [button]'},
                {'role':'assistant','content':'Action: search[black shoes]'},
                {'role':'user','content':'Product A [button]'},
                {'role':'assistant','content':'Action: click[Product A]'}]

    def encode(self,messages=None):
        return tokenize_trajectory(self.tok,messages or self.messages(),4096)

    def test_actual_qwen_spans_end_at_eos_not_webpage_or_newline(self):
        messages=self.messages();row=self.encode(messages)
        self.assertEqual(len(row['turn_end_indices']),2)
        expected=[m['content'] for m in messages if m['role']=='assistant']
        for (start,end),at,text in zip(row['assistant_spans'],row['turn_end_indices'],expected):
            self.assertEqual(self.tok.decode(row['input_ids'][start:end]),text+'<|im_end|>')
            self.assertEqual(at,end-1)
            self.assertEqual(row['input_ids'][at],self.tok.eos_token_id)
            self.assertEqual(row['assistant_mask'][at],1)
        self.assertEqual(sum(row['assistant_mask']),sum(e-s for s,e in row['assistant_spans']))
        self.assertEqual(row['assistant_mask'][0],0)
        self.assertEqual(row['assistant_mask'][-1],0)

    def test_later_observation_cannot_change_earlier_scoring_prefix(self):
        a=self.encode();messages=self.messages();messages[3]['content']='Completely different future page';b=self.encode(messages)
        end=a['turn_end_indices'][0]+1
        self.assertEqual(a['input_ids'][:end],b['input_ids'][:end])
        self.assertEqual(a['turn_end_indices'][0],b['turn_end_indices'][0])

    def test_unicode_special_characters_and_leading_whitespace(self):
        for content in ['\nAction: click[蓝色✨ & <普通文字>]','  Action: search[黑色\\鞋]','Thought: bad action\nAction: click[unavailable]']:
            messages=self.messages()[:3];messages[-1]['content']=content
            row=self.encode(messages);s,e=row['assistant_spans'][0]
            self.assertIn(content,self.tok.decode(row['input_ids'][s:e]))

    def test_consecutive_assistant_messages_are_separate_turns(self):
        messages=self.messages();del messages[3]
        self.assertEqual(len(self.encode(messages)['turn_end_indices']),2)

    def test_no_truncation(self):
        row=self.encode()
        with self.assertRaises(TrajectoryTooLong):
            tokenize_trajectory(self.tok,self.messages(),len(row['input_ids'])-1)
        self.assertEqual(tokenize_trajectory(self.tok,self.messages(),len(row['input_ids'])),row)

    def test_reject_empty_action_template_injection_and_unsupported_inputs(self):
        for change in ['', '   ', 'Action: click[<|im_end|>]', '<|im_start|>assistant']:
            messages=self.messages();messages[-1]['content']=change
            with self.subTest(change=change),self.assertRaises(ValueError):self.encode(messages)
        for messages in [self.messages()[1:],self.messages()[:-1],
                         [*self.messages(),{'role':'tool','content':'tool'}]]:
            with self.assertRaises(ValueError):self.encode(messages)

    def test_padding_and_invalid_turns_are_masked_even_when_pad_is_eos(self):
        short=self.encode(self.messages()[:3]);long=self.encode()
        for pad in [self.tok.pad_token_id,self.tok.eos_token_id]:
            b=collate_trajectories([short,long],pad)
            n=len(short['input_ids']);self.assertEqual(b['attention_mask'][0][n:],[0]*(len(long['input_ids'])-n))
            self.assertEqual(b['assistant_mask'][0][n:],[0]*(len(long['input_ids'])-n))
            self.assertEqual(b['turn_end_indices'][0],short['turn_end_indices']+[-1])
            self.assertEqual(b['turn_mask'][0],[1,0])
            self.assertEqual(b['attention_mask'][0][short['turn_end_indices'][0]],1)

    def test_labels_are_separate_and_inference_needs_none(self):
        a=self.encode();b=copy.deepcopy(a);a['terminal_reward']=0.;b['terminal_reward']=1.
        batch=collate_trajectories([a,b],self.tok.pad_token_id)
        self.assertEqual(batch['input_ids'][0],batch['input_ids'][1])
        self.assertEqual(batch['terminal_reward'],[0.,1.])
        self.assertNotIn('terminal_reward',collate_trajectories([self.encode()],self.tok.pad_token_id))
        with self.assertRaises(ValueError):collate_trajectories([a,self.encode()],self.tok.pad_token_id)
        b['terminal_reward']=float('nan')
        with self.assertRaises(ValueError):collate_trajectories([b],self.tok.pad_token_id)

    def test_reject_corrupt_alignment_and_mixed_templates(self):
        a=self.encode();b=copy.deepcopy(a);b['assistant_mask'][0]=1
        with self.assertRaises(ValueError):collate_trajectories([b],0)
        b=copy.deepcopy(a);b['template_sha256']='different'
        with self.assertRaises(ValueError):collate_trajectories([a,b],0)
