"""Timing accounting, balanced workloads, padding tokens, and selection gates."""
import copy
import json
import tempfile
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from benchmark_utils import choose_batch, percentile, scheduled_batch, summarize, trim_after_eos, workload
from verify_step6 import verify_timings


class BenchmarkTests(unittest.TestCase):
    def test_balanced_reproducible_stream_and_wrapping(self):
        rows = [dict(path=f'{label}/{i}', label=label) for label in ['glioma','meningioma','notumor','pituitary'] for i in range(40)]
        pool = workload(rows, 32, 42)
        self.assertEqual(len({r['path'] for r in pool}), 128)
        self.assertEqual(pool, workload(rows, 32, 42))
        self.assertNotEqual(pool, workload(rows, 32, 43))
        self.assertEqual([r['label'] for r in pool[:4]], ['glioma','meningioma','notumor','pituitary'])
        self.assertEqual(scheduled_batch(pool, 4, 32), pool[:4])
        self.assertEqual([r for i in range(2) for r in scheduled_batch(pool, 2, i)], scheduled_batch(pool, 4, 0))

    def test_eos_trimming_removes_only_post_completion_padding(self):
        self.assertEqual(trim_after_eos([10,11,99,0,0], 99), [10,11,99])
        self.assertEqual(trim_after_eos([10,98,0], [98,99]), [10,98])
        self.assertEqual(trim_after_eos([10,0,11], 99), [10,0,11])

    def test_timing_excludes_warmup_and_uses_ratio_of_totals(self):
        rows = [dict(phase='warmup', elapsed_ns=99_000_000, images=2, trial=0, peak_allocated_bytes=999, peak_reserved_bytes=999)]
        rows += [dict(phase='measured', elapsed_ns=n*1_000_000, images=2, trial=i, peak_allocated_bytes=100+i, peak_reserved_bytes=200+i) for i,n in enumerate([10,30])]
        out = summarize(rows)
        self.assertEqual(out['measurements'], 2)
        self.assertEqual(out['mean_batch_ms'], 20)
        self.assertEqual(out['mean_amortized_image_ms'], 10)
        self.assertEqual(out['throughput_images_s'], 100)
        self.assertEqual(out['p95_batch_ms'], 29)
        self.assertEqual(out['peak_allocated_bytes'], 101)
        with self.assertRaises(ValueError):
            summarize([dict(rows[-1], elapsed_ns=0)])
        with self.assertRaises(ValueError):
            summarize([rows[-1], dict(rows[-1], images=1)])

    def test_percentile_interpolation_and_single_sample(self):
        self.assertEqual(percentile([1,2,3,4], .5), 2.5)
        self.assertEqual(percentile([7], .95), 7)

    def test_selection_respects_oom_memory_and_throughput_plateau(self):
        candidates = [dict(batch_size=b, status='passed', total_memory_bytes=1000,
                           summary=dict(throughput_images_s=t, peak_allocated_bytes=m)) for b,t,m in [(1,10,100),(2,19.5,200),(4,20,400),(8,30,900)]]
        candidates.append(dict(batch_size=16, status='oom'))
        self.assertEqual(choose_batch(candidates, .8, .05), 2)
        candidates[1]['summary']['throughput_images_s'] = 18
        self.assertEqual(choose_batch(candidates, .8, .05), 4)
        with self.assertRaises(ValueError):
            choose_batch([dict(batch_size=1,status='oom')], .8, .05)


    def test_independent_timing_verifier_rejects_tampered_times_and_workloads(self):
        cfg=json.loads((Path(__file__).parents[1]/'configs/step6.json').read_text())
        rows=[dict(path=f'{label}/{i}',label=label,sha256=str(i),group_id=str(i))
              for label in cfg['labels'] for i in range(40)]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); records=[]
            for trial,seed in enumerate(cfg['benchmark']['sweep_seeds']):
                pool=workload(rows,32,seed)
                (root/f'workload-{seed}.json').write_text(json.dumps(pool))
                for phase in ['warmup','measured']:
                    for i in range(10):
                        records.append(dict(phase=phase,trial=trial,seed=seed,iteration=i,images=2,
                            elapsed_ns=10_000_000,peak_allocated_bytes=100,peak_reserved_bytes=200,
                            paths=[r['path'] for r in scheduled_batch(pool,2,i)],input_tokens=[1181,1181],
                            generated_token_counts=[2,2],predictions=[dict(raw_response='glioma',prediction='glioma',generated_token_ids=[10,99]) for _ in range(2)]))
            expected=dict(batch_size=2,measurements=20,images=40,mean_batch_ms=10,p50_batch_ms=10,p95_batch_ms=10,
                          mean_amortized_image_ms=5,p50_amortized_image_ms=5,p95_amortized_image_ms=5,
                          throughput_images_s=200,peak_allocated_bytes=100,peak_reserved_bytes=200,
                          trials=[dict(trial=t,samples=10,mean_batch_ms=10,throughput_images_s=200) for t in range(2)],
                          trial_mean_batch_ms_std=0)
            (root/'summary.json').write_text(json.dumps(expected))
            (root/'run.json').write_text(json.dumps(dict(batch_size=2,total_memory_bytes=1000,summary=expected)))
            def save(items):
                (root/'timings.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in items))
            save(records)
            self.assertEqual(verify_timings(root,cfg,rows,'sweep')['throughput_images_s'],200)
            altered=copy.deepcopy(records); altered[0]['elapsed_ns']=900_000_000; save(altered)
            verify_timings(root,cfg,rows,'sweep')  # Warm-up time must not affect measured statistics.
            for key,value in [('elapsed_ns',20_000_000),('paths',['wrong','wrong']),('images',1)]:
                altered=copy.deepcopy(records); altered[-1][key]=value; save(altered)
                with self.assertRaises(ValueError):
                    verify_timings(root,cfg,rows,'sweep')
            save(records[:-1])
            with self.assertRaises(ValueError):
                verify_timings(root,cfg,rows,'sweep')
            # Nanosecond measurements exercise different floating-point summation orders.
            varied=copy.deepcopy(records)
            for i,r in enumerate(varied):
                r['elapsed_ns']=123456789+i*9876543
            stats=summarize(varied)
            (root/'summary.json').write_text(json.dumps(stats))
            (root/'run.json').write_text(json.dumps(dict(batch_size=2,total_memory_bytes=1000,summary=stats)))
            save(varied)
            verify_timings(root,cfg,rows,'sweep')

    def test_fixed_checkpoint_and_inference_protocol(self):
        root=Path(__file__).parents[1]
        cfg=json.loads((root/'configs/step6.json').read_text())
        previous=json.loads((root/'configs/step4.json').read_text())
        self.assertEqual({k:cfg[k] for k in previous if k not in ('stage','batch_size')},
                         {k:v for k,v in previous.items() if k not in ('stage','batch_size')})
        self.assertEqual(cfg['benchmark']['warmup_batches'],10)
        self.assertEqual(cfg['benchmark']['baseline_measured_batches'],50)
        self.assertEqual(len(cfg['benchmark']['baseline_seeds']),3)
        self.assertEqual(cfg['checkpoint_artifact_digest'],json.loads((root/'experiments/step5/runs/13927137/model-artifacts.json').read_text())['artifact_digest'])


if __name__ == '__main__':
    unittest.main()
