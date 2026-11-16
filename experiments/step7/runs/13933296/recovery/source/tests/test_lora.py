"""Verify loss masks, complete training coverage, checkpoint selection, and merge gates."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from lora_utils import choose_checkpoint, epoch_order, merge_passes, supervised_sequences


class LoraTests(unittest.TestCase):
    def test_masks_prompt_and_padding_but_keeps_target_and_eos(self):
        ids, masks, labels = supervised_sequences([[10, 20, 30], [40]], [[5, 2], [6, 7, 2]], 2)
        self.assertEqual(ids, [[10, 20, 30, 5, 2], [40, 6, 7, 2, 2]])
        self.assertEqual(masks, [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
        self.assertEqual(labels, [[-100, -100, -100, 5, 2], [-100, 6, 7, 2, -100]])

    def test_rejects_missing_prompt_or_target(self):
        for prefixes, targets in [([], []), ([[1]], [[]]), ([[]], [[2]]), ([[1]], [[2], [3]])]:
            with self.assertRaises(ValueError):
                supervised_sequences(prefixes, targets, 0)

    def test_every_training_image_once_with_tail_kept(self):
        order = epoch_order(3279, 42, 1)
        self.assertEqual(sorted(order), list(range(3279)))
        self.assertEqual(order, epoch_order(3279, 42, 1))
        self.assertNotEqual(order, epoch_order(3279, 42, 2))
        windows = [order[i:i + 16] for i in range(0, len(order), 16)]
        self.assertEqual(len(windows), 205)
        self.assertEqual(len(windows[-1]), 15)

    def test_selection_uses_f1_then_loss_then_earliest_epoch(self):
        rows = [dict(epoch=1, metrics=dict(n=579, macro_f1=.8), validation_loss=.3),
                dict(epoch=2, metrics=dict(n=579, macro_f1=.8), validation_loss=.2),
                dict(epoch=3, metrics=dict(n=579, macro_f1=.8), validation_loss=.2)]
        self.assertEqual(choose_checkpoint(rows)['epoch'], 2)
        rows[0]['metrics']['macro_f1'] = .81
        self.assertEqual(choose_checkpoint(rows)['epoch'], 1)
        rows[0]['metrics']['n'] = 578
        with self.assertRaises(ValueError):
            choose_checkpoint(rows)

    def test_merge_gates_reject_numeric_or_generation_regressions(self):
        cfg = json.loads((Path(__file__).parents[1] / 'configs/step5.json').read_text())['merge_verification']
        good = dict(probe_count=8, probe_token_matches=8, max_absolute_logit_difference=.01,
                    mean_absolute_logit_difference=.001, full_validation_label_agreement=1.0,
                    absolute_macro_f1_difference=0.0, all_parameters_fp16=True, remaining_lora_modules=0)
        self.assertTrue(merge_passes(cfg, good))
        for field, value in [('probe_token_matches', 7), ('max_absolute_logit_difference', .2),
                             ('mean_absolute_logit_difference', .03), ('full_validation_label_agreement', .98),
                             ('absolute_macro_f1_difference', .01), ('remaining_lora_modules', 1),
                             ('all_parameters_fp16', False)]:
            bad = copy.deepcopy(good)
            bad[field] = value
            self.assertFalse(merge_passes(cfg, bad), field)

    def test_inference_protocol_and_frozen_scope(self):
        repo = Path(__file__).parents[1]
        previous = json.loads((repo / 'configs/step4.json').read_text())
        current = json.loads((repo / 'configs/step5.json').read_text())
        self.assertEqual({k: current[k] for k in previous}, {**previous, 'stage': 'step5'})
        self.assertEqual(current['training']['micro_batch_size'] * current['training']['gradient_accumulation_steps'], 16)
        import re
        pattern = current['training']['target_modules']
        self.assertTrue(re.fullmatch(pattern, 'model.text_model.layers.0.self_attn.q_proj'))
        self.assertFalse(re.fullmatch(pattern, 'model.vision_model.encoder.layers.0.self_attn.q_proj'))


if __name__ == '__main__':
    unittest.main()
