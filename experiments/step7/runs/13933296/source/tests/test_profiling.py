"""Guard against double-counted stages, overlap inflation, and changed answers."""
import copy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from profiling_utils import stage_times,summarize_stages,trace_summary,union_duration,nvidia_uuid
from verify_step7 import output_equal


class ProfilingTests(unittest.TestCase):
    def test_nvidia_uuid_handles_torch_and_nvidia_formats(self):
        value='15c6a3e6-9de0-3c44-211f-b43e257b5a5c'
        self.assertEqual(nvidia_uuid(value),'GPU-'+value)
        self.assertEqual(nvidia_uuid('GPU-'+value),'GPU-'+value)

    def test_nested_vision_is_removed_from_prefill(self):
        spans=[dict(name=k,wall_ns=v,device_ms=0) for k,v in
               [('preprocessing',10),('processor',8),('transfer',5),('vision',40),('first_forward',70),('decode',10),('postprocessing',2)]]
        result=stage_times(spans,100)['wall_ns']
        self.assertEqual(result,dict(preprocessing=10,transfer=5,vision=40,prefill=30,decode=10,other=5))
        self.assertEqual(sum(result.values()),100)

    def test_impossible_accounting_rejected(self):
        with self.assertRaises(ValueError):
            stage_times([dict(name='vision',wall_ns=40)],100)
        with self.assertRaises(ValueError):
            stage_times([dict(name='preprocessing',wall_ns=101)],100)
        with self.assertRaises(ValueError):
            stage_times([dict(name='decode',wall_ns=-1)],100)

    def test_overlapping_gpu_intervals_count_once(self):
        self.assertEqual(union_duration([(1,5),(2,3),(4,7),(9,10)]),7)
        with self.assertRaises(ValueError):
            union_duration([(3,2)])

    def test_trace_clips_activity_to_batches(self):
        trace=dict(traceEvents=[
            dict(name='mri/batch',ph='X',ts=10,dur=10),
            dict(name='kernel-a',cat='kernel',ph='X',ts=5,dur=10),
            dict(name='kernel-b',cat='kernel',ph='X',ts=13,dur=4),
            dict(name='copy',cat='gpu_memcpy',ph='X',ts=18,dur=10)])
        out=trace_summary(trace)
        self.assertEqual(out['gpu_kernel_busy_us'],7)
        self.assertEqual(out['gpu_activity_busy_us'],9)
        self.assertEqual(out['gpu_activity_fraction'],.9)

    def test_trace_requires_cuda_and_batch_annotations(self):
        for events in ([],[dict(cat='kernel',name='x',ph='X',ts=0,dur=1)],
                       [dict(name='mri/batch',ph='X',ts=0,dur=1)]):
            with self.assertRaises(ValueError):
                trace_summary(dict(traceEvents=events))

    def test_summary_excludes_warmup_and_preserves_residual(self):
        records=[]
        for phase,multiplier in [('warmup',99),('measured',1),('measured',2)]:
            stages=stage_times([dict(name='preprocessing',wall_ns=10*multiplier)],20*multiplier)
            records.append(dict(phase=phase,elapsed_ns=20*multiplier,stages=stages,
                                peak_allocated_bytes=1,peak_reserved_bytes=2))
        out=summarize_stages(records)
        self.assertEqual(out['measurements'],2)
        self.assertAlmostEqual(out['stages']['preprocessing']['fraction'],.5)
        self.assertAlmostEqual(out['stages']['other']['fraction'],.5)

    def test_exact_output_gate_rejects_tokens_and_inputs(self):
        a=dict(paths=['x'],input_tokens=[3],generated_token_counts=[2],
               predictions=[dict(raw_response='glioma',prediction='glioma',generated_token_ids=[1,9])])
        output_equal(a,copy.deepcopy(a))
        b=copy.deepcopy(a);b['predictions'][0]['generated_token_ids']=[2,9]
        with self.assertRaises(ValueError):
            output_equal(a,b)
        b=copy.deepcopy(a);b['paths']=['y']
        with self.assertRaises(ValueError):
            output_equal(a,b)

    def test_profile_keeps_frozen_batch_and_three_trials(self):
        root=Path(__file__).parents[1]
        cfg=json.loads((root/'configs/step7.json').read_text())
        baseline=json.loads((root/cfg['reference_run']/'frozen-baseline.json').read_text())
        self.assertEqual(cfg['batch_size'],baseline['batch_size'])
        self.assertEqual(cfg['seeds'],baseline['config']['benchmark']['baseline_seeds'])
        self.assertEqual(cfg['warmup_batches'],10)


if __name__=='__main__':
    unittest.main()
