"""Pure workload, timing-summary, and batch-selection rules for Step 6."""
import math
import random
import statistics

LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']


def workload(rows, per_class, seed):
    rng = random.Random(seed)
    groups = []
    for label in LABELS:
        candidates = sorted((r for r in rows if r['label'] == label), key=lambda r: r['path'])
        groups.append(rng.sample(candidates, per_class))
    return [group[i] for i in range(per_class) for group in groups]


def scheduled_batch(pool, batch_size, iteration):
    if batch_size < 1 or iteration < 0 or not pool:
        raise ValueError('Invalid workload request')
    return [pool[(iteration * batch_size + i) % len(pool)] for i in range(batch_size)]


def trim_after_eos(ids, eos):
    eos = {eos} if isinstance(eos, int) else set(eos)
    for i, token in enumerate(ids):
        if token in eos:
            return ids[:i + 1]
    return ids


def percentile(values, probability):
    if not values or not 0 <= probability <= 1:
        raise ValueError('Invalid percentile')
    values = sorted(values)
    x = (len(values) - 1) * probability
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] + (values[hi] - values[lo]) * (x - lo)


def summarize(records):
    rows = [r for r in records if r['phase'] == 'measured']
    if not rows or any(r['elapsed_ns'] <= 0 or r['images'] <= 0 for r in rows):
        raise ValueError('Missing or invalid measured samples')
    batch_sizes = {r['images'] for r in rows}
    if len(batch_sizes) != 1:
        raise ValueError('Timing summaries require fixed-size batches')
    batch = next(iter(batch_sizes))
    ms = [r['elapsed_ns'] / 1e6 for r in rows]
    trial_ids = sorted({r['trial'] for r in rows})
    trials = []
    for trial in trial_ids:
        samples = [r for r in rows if r['trial'] == trial]
        trials.append(dict(trial=trial, samples=len(samples),
                           mean_batch_ms=statistics.mean(r['elapsed_ns'] / 1e6 for r in samples),
                           throughput_images_s=sum(r['images'] for r in samples) / (sum(r['elapsed_ns'] for r in samples) / 1e9)))
    return dict(batch_size=batch, measurements=len(rows), images=sum(r['images'] for r in rows),
                mean_batch_ms=statistics.mean(ms), p50_batch_ms=percentile(ms, .5), p95_batch_ms=percentile(ms, .95),
                mean_amortized_image_ms=statistics.mean(ms) / batch,
                p50_amortized_image_ms=percentile(ms, .5) / batch, p95_amortized_image_ms=percentile(ms, .95) / batch,
                throughput_images_s=sum(r['images'] for r in rows) / (sum(r['elapsed_ns'] for r in rows) / 1e9),
                peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),
                peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in rows),
                trials=trials,
                trial_mean_batch_ms_std=statistics.stdev(t['mean_batch_ms'] for t in trials) if len(trials) > 1 else 0.0)


def choose_batch(candidates, memory_fraction, throughput_tolerance):
    eligible = [c for c in candidates if c['status'] == 'passed'
                and c['summary']['peak_allocated_bytes'] <= memory_fraction * c['total_memory_bytes']]
    if not eligible:
        raise ValueError('No candidate meets the memory gate')
    best = max(c['summary']['throughput_images_s'] for c in eligible)
    near_best = [c for c in eligible if c['summary']['throughput_images_s'] >= best * (1 - throughput_tolerance)]
    return min(c['batch_size'] for c in near_best)
