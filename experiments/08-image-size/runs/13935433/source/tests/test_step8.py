"""Guard against changing the comparison or hiding a negative optimization result."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark import configure_processor
from step8_utils import comparison, inference_config

ROOT=Path(__file__).resolve().parents[1]


class ImageBudgetTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((ROOT/'configs/step8.json').read_text())
        self.base=json.loads((ROOT/'configs/step6.json').read_text())

    def test_only_one_inference_factor_changes(self):
        a=inference_config(self.base,self.cfg,'baseline',43)
        b=inference_config(self.base,self.cfg,'image1024',43)
        self.assertEqual({k for k in a if a[k]!=b[k]}, {'processor_longest_edge'})
        self.assertEqual(a['benchmark']['baseline_seeds'],[43])
        self.assertNotIn('processor_longest_edge',self.base)
        changed=deepcopy(self.cfg);changed['batch_size']=8
        with self.assertRaises(ValueError):inference_config(self.base,changed,'baseline')

    def test_no_override_preserves_historical_processor(self):
        p=SimpleNamespace(image_processor=SimpleNamespace(size={'longest_edge':2048},max_image_size={'longest_edge':512},do_image_splitting=True))
        configure_processor(p,{})
        self.assertEqual(p.image_processor.size,{'longest_edge':2048})
        configure_processor(p,{'processor_longest_edge':1024})
        self.assertEqual(p.image_processor.size,{'longest_edge':1024})
        with self.assertRaises(ValueError):configure_processor(p,{'processor_longest_edge':1024})

    def test_fast_but_bad_quality_is_not_success(self):
        a=dict(mean_batch_ms=1000,peak_allocated_bytes=300)
        b=dict(mean_batch_ms=500,peak_allocated_bytes=100)
        result=comparison(a,b,.85,dict(macro_f1=.85),dict(macro_f1=.82,invalid_count=0),self.cfg)
        self.assertEqual(result['speedup'],2)
        self.assertFalse(result['preliminary_success'])
        self.assertFalse(result['quality_gate_passed'])

    def test_invalid_output_and_slowdown_remain_visible(self):
        a=dict(mean_batch_ms=1000,peak_allocated_bytes=300)
        b=dict(mean_batch_ms=1100,peak_allocated_bytes=100)
        result=comparison(a,b,.85,dict(macro_f1=.85),dict(macro_f1=.85,invalid_count=0),self.cfg)
        self.assertTrue(result['quality_gate_passed']);self.assertFalse(result['preliminary_success'])
        result=comparison(a,b,.85,dict(macro_f1=.85),dict(macro_f1=.85,invalid_count=1),self.cfg)
        self.assertFalse(result['quality_gate_passed'])


if __name__=='__main__':unittest.main()
