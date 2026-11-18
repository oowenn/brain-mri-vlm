"""Pure accounting for exclusive stage times and profiler GPU activity."""
import statistics

PRIMARY = ['preprocessing', 'transfer', 'vision', 'prefill', 'decode']


def stage_times(spans, elapsed_ns):
    wall = {}
    device = {}
    for item in spans:
        if item['wall_ns'] < 0 or item.get('device_ms', 0) < 0:
            raise ValueError('Negative stage duration')
        wall[item['name']] = wall.get(item['name'], 0) + item['wall_ns']
        device[item['name']] = device.get(item['name'], 0) + item.get('device_ms', 0)
    if wall.get('first_forward', 0) < wall.get('vision', 0):
        raise ValueError('Vision must be nested inside the first forward')
    result = {k:wall.get(k, 0) for k in PRIMARY}
    result['prefill'] = wall.get('first_forward', 0) - wall.get('vision', 0)
    result['other'] = elapsed_ns - sum(result.values())
    if result['other'] < 0:
        raise ValueError('Stage durations exceed the whole batch')
    device['prefill'] = device.get('first_forward', 0) - device.get('vision', 0)
    if device['prefill'] < -0.1:
        raise ValueError('Invalid nested CUDA-event timing')
    return dict(wall_ns=result, device_ms={k:device.get(k, 0) for k in PRIMARY})


def union_duration(intervals):
    total = 0
    end = None
    for a,b in sorted(intervals):
        if b < a:
            raise ValueError('Reversed interval')
        if end is None or a >= end:
            total += b-a
        elif b > end:
            total += b-end
        end = b if end is None else max(end,b)
    return total


def trace_summary(trace):
    events=trace['traceEvents']
    windows=[(e['ts'],e['ts']+e['dur']) for e in events
             if e.get('name')=='mri/batch' and e.get('ph')=='X']
    kernels=[e for e in events if e.get('cat')=='kernel' and e.get('ph')=='X']
    copies=[e for e in events if e.get('cat') in ('gpu_memcpy','gpu_memset') and e.get('ph')=='X']
    if not windows or not kernels:
        raise ValueError('Trace needs annotated batch windows and real CUDA kernels')
    def clipped(items):
        return [(max(a,e['ts']),min(b,e['ts']+e['dur'])) for a,b in windows for e in items
                if max(a,e['ts']) < min(b,e['ts']+e['dur'])]
    window_us=union_duration(windows)
    kernel_us=union_duration(clipped(kernels))
    activity_us=union_duration(clipped(kernels+copies))
    names={}
    for e in kernels:
        names[e['name']]=names.get(e['name'],0)+e['dur']
    return dict(batch_windows=len(windows),batch_window_us=window_us,
                kernel_events=len(kernels),copy_or_memset_events=len(copies),
                gpu_kernel_busy_us=kernel_us,gpu_activity_busy_us=activity_us,
                gpu_activity_fraction=activity_us/window_us,
                top_kernels=[dict(name=k,total_us=v) for k,v in sorted(names.items(),key=lambda p:-p[1])[:30]],
                annotation_names=sorted({e.get('name') for e in events if str(e.get('name','')).startswith('mri/')}))


def summarize_stages(records):
    measured=[r for r in records if r['phase']=='measured']
    if not measured:
        raise ValueError('No measured profiles')
    total=sum(r['elapsed_ns'] for r in measured)
    stages={}
    for name in PRIMARY+['other']:
        values=[r['stages']['wall_ns'][name]/1e6 for r in measured]
        stages[name]=dict(mean_ms=statistics.mean(values),
                          p50_ms=statistics.median(values),
                          p95_ms=statistics.quantiles(values,n=100,method='inclusive')[94],
                          fraction=sum(values)/(total/1e6))
    return dict(measurements=len(measured),mean_batch_ms=total/len(measured)/1e6,
                stages=stages,
                dominant_stage=max(PRIMARY,key=lambda n:stages[n]['mean_ms']),
                peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in measured),
                peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in measured))
