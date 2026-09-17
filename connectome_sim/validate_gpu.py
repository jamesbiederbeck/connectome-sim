"""Tier-2 validation: cross-check GPUBrain against NativeBrain on the real graph.

Tier 1 (tests/test_doom_reference.py) already certifies GPUBrain exactly against
the independent Brian2 oracle on a tiny graph. This script empirically bounds
the floating-point-accumulation-order concern doc/doom-performance-review.md
raises at the real 166,700-neuron/25.58M-edge scale: it runs both backends from
identical initial state on an identical synthetic luminance sequence, and
reports (a) any per-neuron spike-count mismatch, and (b) how often either
backend's voltage sits within a small epsilon of the -45mV threshold at
spike-check time, so a mismatch (if any) can be judged against how often the
system is even in a summation-order-sensitive regime, rather than asserted
away in either direction.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from doom.native import NativeBrain
from doom.gpu import GPUBrain

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD_MV = -45.0

def run(ticks, dataset, seed, near_threshold_eps):
    path = ROOT / 'outputs/doom' / dataset / 'graph.npz'
    native, gpu = NativeBrain(path), GPUBrain(path)
    n = native.n
    rng = np.random.default_rng(seed)
    total_counts = {'native': np.zeros(n, dtype=np.int64), 'gpu': np.zeros(n, dtype=np.int64)}
    near_threshold_total = {'native': 0, 'gpu': 0}
    first_mismatch_tick = None
    history = []
    cursor = 0
    for tick in range(1, ticks + 1):
        target = int(round(tick * 10000 / 35))
        steps = target - cursor; cursor = target
        light = rng.uniform(0, 1, size=len(native.retina)).astype(np.float32)
        nc, _ = native.step(light, steps * .1)
        gc, _ = gpu.step(light, steps * .1)
        total_counts['native'] += nc
        total_counts['gpu'] += gc
        v_gpu_host = gpu.v.get()
        near_native = int(np.count_nonzero(np.abs(native.v - THRESHOLD_MV) < near_threshold_eps))
        near_gpu = int(np.count_nonzero(np.abs(v_gpu_host - THRESHOLD_MV) < near_threshold_eps))
        near_threshold_total['native'] += near_native
        near_threshold_total['gpu'] += near_gpu
        cumulative_mismatch = int(np.count_nonzero(total_counts['native'] != total_counts['gpu']))
        if first_mismatch_tick is None and cumulative_mismatch > 0: first_mismatch_tick = tick
        history.append({
            'tick': tick, 'cumulative_mismatched_neurons': cumulative_mismatch,
            'near_threshold_events_this_tick': {'native': near_native, 'gpu': near_gpu},
            'voltage_max_abs_diff_mv': float(np.abs(native.v - v_gpu_host).max()),
        })
    mismatch = total_counts['native'] != total_counts['gpu']
    v_native, v_gpu = native.v, gpu.v.get()
    g_native, g_gpu = native.g, gpu.g.get()
    report = {
        'dataset': dataset, 'ticks': ticks, 'seed': seed, 'neurons': int(n),
        'simulated_ms': native.sim_ms,
        'spike_count_mismatch_neurons': int(mismatch.sum()),
        'spike_count_total': {'native': int(total_counts['native'].sum()), 'gpu': int(total_counts['gpu'].sum())},
        'spike_count_max_abs_diff': int(np.abs(total_counts['native'].astype(np.int64) - total_counts['gpu'].astype(np.int64)).max()),
        'near_threshold_epsilon_mv': near_threshold_eps,
        'near_threshold_events_total': near_threshold_total,
        'first_mismatch_tick': first_mismatch_tick,
        'final_voltage_max_abs_diff_mv': float(np.abs(v_native - v_gpu).max()),
        'final_conductance_max_abs_diff_mv': float(np.abs(g_native - g_gpu).max()),
        'history': history,
    }
    return report

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ticks', type=int, default=200)
    p.add_argument('--dataset', default='malecns_v1')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--near-threshold-eps', type=float, default=1e-3)
    p.add_argument('--out')
    args = p.parse_args()
    start = time.time()
    report = run(args.ticks, args.dataset, args.seed, args.near_threshold_eps)
    report['wall_seconds'] = round(time.time() - start, 3)
    print(json.dumps(report, indent=2))
    if args.out:
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + '\n')

if __name__ == '__main__':
    main()
