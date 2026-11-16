"""Fixed one-factor image-budget protocol and interpretation of negative results."""
from copy import deepcopy


def inference_config(baseline, cfg, variant, seed=None):
    if cfg['batch_size'] != 4 or cfg['seeds'] != [42, 43, 44]:
        raise ValueError('Step 8 must retain the frozen batch and seeds')
    if cfg['longest_edges'] != {'baseline': 2048, 'image1024': 1024}:
        raise ValueError('Only the declared image-budget comparison is supported')
    if cfg['trial_orders'] != [['baseline', 'image1024'], ['image1024', 'baseline'], ['baseline', 'image1024']]:
        raise ValueError('Paired trial order changed')
    if cfg['warmup_batches'] != 10 or cfg['measured_batches'] != 50 or cfg['stage_batches_per_trial'] != 10:
        raise ValueError('Frozen measurement counts changed')
    if cfg['quality_max_macro_f1_drop'] != 0.02 or cfg['quality_max_invalid_count'] != 0:
        raise ValueError('Quality gate changed')
    if seed is not None and seed not in cfg['seeds']:
        raise ValueError('Unknown timing seed')
    result = deepcopy(baseline)
    result['processor_longest_edge'] = cfg['longest_edges'][variant]
    if seed is not None:
        result['benchmark']['baseline_seeds'] = [seed]
    return result


def comparison(baseline, variant, reference_f1, baseline_quality, variant_quality, cfg):
    floor = reference_f1 - cfg['quality_max_macro_f1_drop']
    quality_pass = (variant_quality['macro_f1'] >= floor and
                    baseline_quality['macro_f1'] - variant_quality['macro_f1'] <= cfg['quality_max_macro_f1_drop'] and
                    variant_quality['invalid_count'] <= cfg['quality_max_invalid_count'])
    faster = variant['mean_batch_ms'] < baseline['mean_batch_ms']
    return dict(reference_macro_f1=reference_f1, validation_floor=floor,
                quality_gate_passed=quality_pass, faster=faster,
                preliminary_success=quality_pass and faster,
                speedup=baseline['mean_batch_ms']/variant['mean_batch_ms'],
                latency_reduction_fraction=1-variant['mean_batch_ms']/baseline['mean_batch_ms'],
                peak_allocated_reduction_fraction=1-variant['peak_allocated_bytes']/baseline['peak_allocated_bytes'],
                macro_f1_change=variant_quality['macro_f1']-baseline_quality['macro_f1'],
                interpretation='preliminary improvement within validation gate' if quality_pass and faster else
                    'quality gate failed; measured performance is not an accepted optimization' if not quality_pass else
                    'quality gate passed but no mean latency improvement')
