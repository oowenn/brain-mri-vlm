#!/usr/bin/env python3
"""Create the report-facing summary only after both jobs and artifact checks pass."""
import argparse
import csv
import json
from pathlib import Path
import re


def read(path):return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();root=args.run_dir
    record=read(root/'run.json');report=read(root/'independent-verification.json')
    remote=read(root/'wandb-verification.json');receipt=read(root/'evidence-upload.json')
    def job(path,job_id=None):
        with path.open() as stream:rows=list(csv.DictReader(stream,delimiter='|'))
        row=next(r for r in rows if r['JobIDRaw']==job_id) if job_id else next(r for r in rows if r['JobIDRaw'].isdigit())
        if row['State']!='COMPLETED' or row['ExitCode']!='0:0':raise ValueError('Job has not completed successfully')
        return row
    gpu=job(root/'slurm-accounting.tsv',root.name)
    publisher=job(root/'publisher-slurm-accounting.tsv')
    if record['status']!='passed' or report['status']!='passed' or remote['status']!='passed':
        raise ValueError('Verification incomplete')
    if record['test_set_inference'] or report['test_set_inference']:
        raise ValueError('Unexpected test inference')
    commit=(root/'base-commit.txt').read_text().strip()
    if commit!=record['base_commit'] or read(root/'source-provenance.json')['commit']!=commit:
        raise ValueError('Execution source differs')
    tests=(root/'tests.txt').read_text();count=re.search(r'Ran (\d+) tests',tests)
    if not count or not tests.rstrip().endswith('OK'):raise ValueError('Tests did not pass')
    result=dict(report,status='verified-complete',stage='step8',execution_commit=commit,
                gpu_job=gpu,publisher_job=publisher,unit_tests_passed=int(count.group(1)),
                wandb_run_id=record['wandb_run_id'],wandb_url=record['wandb_url'],
                wandb_verification=remote,evidence=receipt,
                optimization_scope='one image-budget technique; preliminary Step 9 measurements complete; other techniques deferred')
    path=root.parents[1]/'summary.json';path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(path)


if __name__=='__main__':main()
