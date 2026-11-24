"""Offline pinned-Qwen check on legacy SFT dialogues; no fitting or GPU work."""
import argparse
import hashlib
import json
from pathlib import Path
from webshop_lab.model import training_messages
from webshop_lab.prm.config import load_config
from webshop_lab.prm.tokenize import TrajectoryTooLong, collate_trajectories, tokenize_trajectory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sft-json',required=True)
    parser.add_argument('--config',default='configs/prm/qwen25-b1-interface-example.json')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    from transformers import AutoTokenizer
    import transformers
    cfg=load_config(args.config)
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',revision=cfg.tokenizer_revision,
                                    local_files_only=True,trust_remote_code=False)
    from huggingface_hub import hf_hub_download
    tokenizer_hashes={}
    for name,expected in cfg.tokenizer_files_sha256.items():
        cached=Path(hf_hub_download('Qwen/Qwen2.5-1.5B-Instruct',name,
            revision=cfg.tokenizer_revision,local_files_only=True))
        actual=hashlib.sha256(cached.read_bytes()).hexdigest()
        if actual!=expected:raise ValueError(f'Pinned tokenizer identity mismatch: {name}')
        tokenizer_hashes[name]=actual
    source=Path(args.sft_json)
    rows=json.loads(source.read_text(encoding='utf-8'))
    records=[];accepted=[]
    # Only inspect a deterministic sample. Never write original dialogue text.
    for index,row in enumerate(rows[:16]):
        messages=training_messages(row)
        while messages and messages[-1]['role']!='assistant':messages.pop()
        try:
            encoded=tokenize_trajectory(tok,messages,cfg.max_length)
        except TrajectoryTooLong as exc:
            records.append({'source_row':index,'status':'rejected_overlength','reason':str(exc)})
            continue
        ends=encoded['turn_end_indices']
        assert len(ends)==sum(m['role']=='assistant' for m in messages)
        assert all(encoded['input_ids'][i]==tok.eos_token_id for i in ends)
        records.append({'source_row':index,'status':'encoded','tokens':len(encoded['input_ids']),
                        'turns':len(ends),'turn_end_indices':ends,'all_score_tokens_are_im_end':True})
        accepted.append(encoded)
    assert accepted,'No sample fits this context; do not claim successful real-data check'
    batch=collate_trajectories(accepted,tok.eos_token_id)
    for row,attention,mask in zip(accepted,batch['attention_mask'],batch['assistant_mask']):
        n=len(row['input_ids']);assert not any(attention[n:]) and not any(mask[n:])
    result={'scope':'CPU tokenizer-only validation, not PRM data fitting or GPU training',
            'transformers':transformers.__version__,'tokenizer_revision':cfg.tokenizer_revision,
            'config_fingerprint':cfg.fingerprint(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'sampled_rows':len(records),'accepted':len(accepted),'rejected_overlength':len(records)-len(accepted),
            'score_position':'assistant_im_end','template_sha256':accepted[0]['template_sha256'],
            'tokenizer_files_sha256':tokenizer_hashes,
            'batch_pad_equals_eos_checked':True,'records':records}
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':main()
