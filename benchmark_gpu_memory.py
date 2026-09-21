"""Standalone tics/sec benchmark: MemoryBrain (CPU) vs GPUMemoryBrain, same
shape as benchmark_gpu.py's existing native-vs-GPU comparison for the plain
kernel. No server/game; synthetic luminance, frozen weights (learning=False
matches MemoryBrain.step()'s always-False call into the native kernel; the
outer rate-bin/rule.advance() bookkeeping runs identically in both regardless
of this flag and its cost is the same on both backends, so it does not bias
the comparison toward either side).
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from connectome_sim.physiology.brain import MemoryBrain
from connectome_sim.physiology.gpu_brain import GPUMemoryBrain
from connectome_sim.physiology.common import ROOT


def _run_one(brain, ticks, warmup, seed):
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
    return {
        'total_wall_seconds': round(total_wall, 4),
        'tics_per_second': round(ticks / total_wall, 3),
        'step_wall_ms': {
            'median': float(np.median(step_wall_ms)), 'mean': float(np.mean(step_wall_ms)),
            'min': float(step_wall_ms.min()), 'max': float(step_wall_ms.max()),
            'p95': float(np.percentile(step_wall_ms, 95)),
        },
    }


def run(ticks, dataset, seed, warmup):
    path = ROOT / 'outputs/connectome_sim' / dataset / 'graph.npz'
    probe = MemoryBrain(path)
    circuit = probe.circuit
    modulation_mask = probe.modulation_mask
    cpu = MemoryBrain(path, circuit=circuit, modulation_mask=modulation_mask)
    gpu = GPUMemoryBrain(path, circuit=circuit, modulation_mask=modulation_mask)
    cpu_report = _run_one(cpu, ticks, warmup, seed)
    gpu_report = _run_one(gpu, ticks, warmup, seed)
    return {
        'dataset': dataset, 'ticks': ticks, 'warmup_ticks': warmup, 'seed': seed,
        'neurons': int(cpu.n), 'edges': int(len(cpu.post)), 'plastic_edges': int(len(circuit['edges'])),
        'hardware': 'measured on this host; see docs/gpu-learning-kernel-review.md',
        'MemoryBrain': cpu_report, 'GPUMemoryBrain': gpu_report,
        'speedup': round(gpu_report['tics_per_second'] / cpu_report['tics_per_second'], 3),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ticks', type=int, default=100)
    p.add_argument('--warmup', type=int, default=10)
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
