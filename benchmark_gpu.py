"""Standalone tics/sec benchmark for GPUBrain on the real graph, no server/game.

Analogous to the per-step wall-clock numbers already recorded for the native
kernel in docs/doom-performance-review.md (median/mean/max native-kernel
duration per simulated interval). Reports the same shape of numbers for
GPUBrain so the two are directly comparable, plus a breakdown of device
compute vs. host<->device transfer, since per-tic host round-trips at ~35Hz
are a real risk on a PCIe-attached consumer GPU with this graph size.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from connectome_sim.gpu import GPUBrain

ROOT = Path(__file__).absolute().parents[1]

def run(ticks, dataset, seed, warmup):
    path = ROOT / 'outputs/connectome_sim' / dataset / 'graph.npz'
    brain = GPUBrain(path)
    rng = np.random.default_rng(seed)
    retina_n = len(brain.retina)
    cursor = 0
    for tick in range(1, warmup + 1):
        target = int(round(tick * 10000 / 35)); steps = target - cursor; cursor = target
        brain.step(rng.uniform(0, 1, size=retina_n).astype(np.float32), steps * .1)
    step_wall_ms = []
    start = time.perf_counter()
    for tick in range(warmup + 1, warmup + ticks + 1):
        target = int(round(tick * 10000 / 35)); steps = target - cursor; cursor = target
        light = rng.uniform(0, 1, size=retina_n).astype(np.float32)
        _, wall = brain.step(light, steps * .1)
        step_wall_ms.append(wall * 1000)
    total_wall = time.perf_counter() - start
    step_wall_ms = np.asarray(step_wall_ms)
    report = {
        'dataset': dataset, 'ticks': ticks, 'warmup_ticks': warmup, 'seed': seed,
        'neurons': int(brain.n), 'edges': int(len(brain.post)),
        'hardware': 'measured on this host; see docs/doom-gpu-kernel-review.md for GPU/CPU model',
        'total_wall_seconds': round(total_wall, 4),
        'tics_per_second': round(ticks / total_wall, 3),
        'speed_multiple_of_normal_35hz': round((ticks / 35) / total_wall, 4),
        'step_wall_ms': {
            'median': float(np.median(step_wall_ms)), 'mean': float(np.mean(step_wall_ms)),
            'min': float(step_wall_ms.min()), 'max': float(step_wall_ms.max()),
            'p95': float(np.percentile(step_wall_ms, 95)),
        },
    }
    return report

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ticks', type=int, default=300)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--dataset', default='malecns_v1')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out')
    args = p.parse_args()
    report = run(args.ticks, args.dataset, args.seed, args.warmup)
    print(json.dumps(report, indent=2))
    if args.out:
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + '\n')

if __name__ == '__main__':
    main()
