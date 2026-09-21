"""Tier 2 validation: cross-check GPUMemoryBrain against MemoryBrain on the
real graph, same synthetic-luminance cross-check methodology validate_gpu.py
already used for the plain kernel (GPUBrain vs NativeBrain).

Runs two phases from identical initial state and an identical circuit
(computed once and passed to both, so this isn't also cross-checking
circuit.identify against itself):

1. Frozen weights (learning=False): reports per-neuron spike-count mismatch
   the way validate_gpu.py's report does for the plain kernel.
2. A short learning run (learning=True): reports the same spike-count
   mismatch, plus the plastic-edge weight/memory_w mismatch this port adds
   on top of the plain kernel's validation -- this is the check that would
   have caught the device-weight-upload bug (see physiology/gpu_brain.py
   module docstring point 4) if it had shipped broken.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from connectome_sim.physiology.brain import MemoryBrain
from connectome_sim.physiology.gpu_brain import GPUMemoryBrain
from connectome_sim.physiology.circuit import identify
from connectome_sim.physiology.common import ROOT


def run(ticks, dataset, seed, learning):
    # ROOT comes from physiology/common.py, which resolves relative to
    # wherever connectome_sim was imported from (e.g. through the
    # androsophila/connectome_sim -> ../connectome-sim symlink, ROOT is
    # androsophila/, where outputs/connectome_sim/malecns_v1/graph.npz and
    # the compiled libneural.so/libmemory.so actually live) -- not
    # necessarily this script's own directory.
    path = ROOT / 'outputs/connectome_sim' / dataset / 'graph.npz'
    probe = MemoryBrain(path)
    circuit = probe.circuit
    modulation_mask = probe.modulation_mask
    cpu = MemoryBrain(path, circuit=circuit, modulation_mask=modulation_mask)
    gpu = GPUMemoryBrain(path, circuit=circuit, modulation_mask=modulation_mask)
    n = cpu.n
    rng = np.random.default_rng(seed)
    total_counts = {'cpu': np.zeros(n, dtype=np.int64), 'gpu': np.zeros(n, dtype=np.int64)}
    first_mismatch_tick = None
    history = []
    cursor = 0
    for tick in range(1, ticks + 1):
        target = int(round(tick * 10000 / 35))
        steps = target - cursor
        cursor = target
        light = rng.uniform(0, 1, size=len(cpu.retina)).astype(np.float32)
        cc, _ = cpu.step(light, steps * .1, learning=learning, lamina_bias=0)
        gc, _ = gpu.step(light, steps * .1, learning=learning, lamina_bias=0)
        total_counts['cpu'] += cc
        total_counts['gpu'] += gc
        cumulative_mismatch = int(np.count_nonzero(total_counts['cpu'] != total_counts['gpu']))
        if first_mismatch_tick is None and cumulative_mismatch > 0:
            first_mismatch_tick = tick
        history.append({'tick': tick, 'cumulative_mismatched_neurons': cumulative_mismatch})
    mismatch = total_counts['cpu'] != total_counts['gpu']
    report = {
        'dataset': dataset, 'ticks': ticks, 'seed': seed, 'neurons': int(n), 'learning': learning,
        'simulated_ms': cpu.sim_ms,
        'spike_count_mismatch_neurons': int(mismatch.sum()),
        'spike_count_mismatch_fraction': float(mismatch.sum()) / n,
        'spike_count_total': {'cpu': int(total_counts['cpu'].sum()), 'gpu': int(total_counts['gpu'].sum())},
        'first_mismatch_tick': first_mismatch_tick,
        'final_voltage_max_abs_diff_mv': float(np.abs(cpu.v - gpu.v.get()).max()),
    }
    if learning:
        report['plastic_edges'] = len(circuit['edges'])
        report['weight_max_abs_diff'] = float(np.abs(cpu.weight[circuit['edges']] - gpu.weight[circuit['edges']]).max())
        report['memory_w_max_abs_diff'] = float(np.abs(cpu.memory_w - gpu.memory_w).max())
        report['changed_edges'] = {'cpu': cpu.memory()['changed_edges'], 'gpu': gpu.memory()['changed_edges']}
    report['history'] = history
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ticks', type=int, default=50)
    p.add_argument('--dataset', default='malecns_v1')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--learning', action='store_true')
    p.add_argument('--out')
    args = p.parse_args()
    start = time.time()
    report = run(args.ticks, args.dataset, args.seed, args.learning)
    report['wall_seconds'] = round(time.time() - start, 3)
    print(json.dumps(report, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
