#!/usr/bin/env python3
"""Summarize all changed validation labels after the verified paired experiment."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();root=args.run_dir
    verification=json.loads((root/'independent-verification.json').read_text())
    if verification['status']!='passed':raise ValueError('Verify execution first')
    names=['baseline-quality/predictions.jsonl','image1024-quality/predictions.jsonl']
    arrays=[[json.loads(line) for line in (root/name).read_text().splitlines()] for name in names]
    if len(arrays[0])!=579 or len(arrays[1])!=579:raise ValueError('Incomplete validation')
    changes=[];transitions=Counter()
    for a,b in zip(*arrays):
        if a['path']!=b['path'] or a['true_label']!=b['true_label']:raise ValueError('Unpaired images')
        if a['prediction']!=b['prediction']:
            change=dict(path=a['path'],true_label=a['true_label'],baseline=a['prediction'],variant=b['prediction'],
                        baseline_correct=a['prediction']==a['true_label'],variant_correct=b['prediction']==b['true_label'])
            changes.append(change);transitions[f"{a['prediction']} -> {b['prediction']}"]+=1
    result=dict(status='derived-from-verified-predictions',images=579,changed_labels=len(changes),
                correct_to_wrong=sum(c['baseline_correct'] and not c['variant_correct'] for c in changes),
                wrong_to_correct=sum(not c['baseline_correct'] and c['variant_correct'] for c in changes),
                wrong_to_different_wrong=sum(not c['baseline_correct'] and not c['variant_correct'] for c in changes),
                class_transitions=dict(transitions),changes=changes,
                input_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names})
    (root/'prediction-changes.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('changes','input_sha256')},indent=2))


if __name__=='__main__':main()
