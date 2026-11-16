"""Independently count trace GPU activity by sweeping interval endpoints."""
import gzip
import json
import math
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'runs/13933296'
checks=[]
for seed in (42,43,44):
    with gzip.open(root/f'traces/trace-{seed}.json.gz','rt') as f:
        events=json.load(f)['traceEvents']
    windows=[(e['ts'],e['ts']+e['dur']) for e in events if e.get('cat')=='user_annotation' and e.get('name')=='mri/batch' and e.get('ph')=='X']
    assert len(windows)==2
    endpoints=[]
    for e in events:
        if e.get('ph')!='X' or e.get('cat') not in ('kernel','gpu_memcpy','gpu_memset'):
            continue
        for a,b in windows:
            lo,hi=max(a,e['ts']),min(b,e['ts']+e['dur'])
            if lo<hi:
                endpoints.extend([(lo,1),(hi,-1)])
    total=0;active=0;previous=None
    for t,delta in sorted(endpoints):
        if active:
            total+=t-previous
        active+=delta;previous=t
        assert active>=0
    assert active==0
    old=json.loads((root/f'traces/trace-{seed}-summary.json').read_text())
    assert math.isclose(total,old['gpu_activity_busy_us'],rel_tol=1e-12)
    checks.append(dict(seed=seed,cpu_batches=2,gpu_activity_us=total,independent_endpoint_sweep_matches=True))
Path(__file__).with_name('trace-interval-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print('PASS: independent endpoint sweeps reproduce all three trace activity totals')
