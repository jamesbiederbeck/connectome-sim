"""Tier 1: toy synthetic graph, GPUMemoryBrain vs MemoryBrain from identical
initial state and identical stimulation, modeled on test_doom_learning_v6.py's
4-neuron graph (plastic edge, kc_mask, dan_index, modulation_mask).

Skipped entirely if no CUDA-capable cupy is importable in this environment --
see docs/gpu-learning-kernel-review.md for why (and for the Tier 2/benchmark/
determinism numbers this suite doesn't attempt to reproduce at toy scale).
"""
import numpy as np
import pytest
from connectome_sim.physiology.brain import MemoryBrain

cp = pytest.importorskip('cupy', reason='GPUMemoryBrain requires cupy + a CUDA device')
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip('No CUDA device visible to cupy', allow_module_level=True)
except Exception:
    pytest.skip('No usable CUDA device for cupy', allow_module_level=True)

from connectome_sim.physiology.gpu_brain import GPUMemoryBrain  # noqa: E402


def _graph(tmp_path):
    p = tmp_path / 'graph.npz'
    n = 4
    np.savez(p, ptr=np.array([0, 1, 1, 2, 2], dtype=np.int64), post=np.array([1, 1], dtype=np.int32),
              weight=np.array([20., .275], dtype=np.float32), ids=np.arange(n, dtype=np.int64),
              retina=np.empty(0, dtype=np.int32), uv=np.empty((0, 2), dtype=np.float32),
              lamina=np.empty(0, dtype=np.int32), sugar=np.empty(0, dtype=np.int32),
              superclass=np.array(['test'] * n))
    return p


def _circuit():
    return {'edges': np.array([0], dtype=np.int64), 'pre': np.array([0], dtype=np.int32),
            'kc_mask': np.array([1, 0, 0, 0], dtype=np.uint8), 'dan_index': np.array([-1, -1, 0, -1], dtype=np.int8),
            'gain': np.array([[1.]], dtype=np.float32), 'kc': np.array([0]), 'mb': np.array([1]), 'dan': np.array([2])}


def _pair(tmp_path):
    p = _graph(tmp_path)
    c = _circuit()
    mask = np.array([0, 0, 1, 0])
    cpu = MemoryBrain(p, eta=.001, circuit=c, modulation_mask=mask)
    gpu = GPUMemoryBrain(p, eta=.001, circuit=c, modulation_mask=mask)
    return cpu, gpu


def test_matches_cpu_spike_counts_no_learning(tmp_path):
    cpu, gpu = _pair(tmp_path)
    for stimulus in [([0], 20), ([2], 20), ([0, 2], 20)]:
        c_cpu, _ = cpu.step([], 100, stimulation=stimulus, lamina_bias=0)
        c_gpu, _ = gpu.step([], 100, stimulation=stimulus, lamina_bias=0)
        np.testing.assert_array_equal(c_cpu, c_gpu)
    # Exact float match is not the bar (kernel.cpp is lazy/table-based, the
    # GPU kernel is dense/closed-form at d=1 -- they differ in float32
    # rounding, per the project's own convention in test_doom_learning_v6.py
    # for the plain kernel: exact spike counts, voltages within atol=.002.
    np.testing.assert_allclose(cpu.v, gpu.v.get(), atol=.002, rtol=0)


def test_modulation_mask_drops_transmitter_delivery_both_backends(tmp_path):
    # Neuron 2 is the sole modulation_mask cell and has a real out-edge to
    # neuron 1 (weight .275). Firing it directly must not raise neuron 1's
    # conductance on either backend.
    cpu, gpu = _pair(tmp_path)
    cpu.step([], 100, stimulation=([2], 20), lamina_bias=0)
    gpu.step([], 100, stimulation=([2], 20), lamina_bias=0)
    assert cpu.g[1] == 0.
    assert float(gpu.g.get()[1]) == 0.


def test_learning_run_updates_weight_on_both_backends_and_device_weight_matches(tmp_path):
    cpu, gpu = _pair(tmp_path)
    cpu.step([], 100, learning=True, stimulation=([0], 20), lamina_bias=0)
    cpu.step([], 100, learning=True, stimulation=([2], 12), lamina_bias=0)
    gpu.step([], 100, learning=True, stimulation=([0], 20), lamina_bias=0)
    gpu.step([], 100, learning=True, stimulation=([2], 12), lamina_bias=0)
    assert cpu.memory_u[0] < 0 and cpu.weight[0] < 20
    assert gpu.memory_u[0] < 0 and gpu.weight[0] < 20
    # The critical regression this guards: the outer loop writes
    # self.weight (host numpy) via rule.advance(), but the kernel reads
    # self._csr_weight (device). Without the scatter in step()/reset(),
    # this assertion fails even though host-side weight/memory_u/memory_w
    # look identical to the CPU run.
    device_weight = float(gpu._csr_weight[int(gpu.circuit['edges'][0])].get())
    assert device_weight == pytest.approx(float(gpu.weight[0]), rel=0, abs=1e-6)
    np.testing.assert_allclose(cpu.weight[0], gpu.weight[0], atol=1e-4)
    np.testing.assert_allclose(cpu.memory_w[0], gpu.memory_w[0], atol=1e-4)


def test_reset_restores_rest_and_baseline_weight(tmp_path):
    _, gpu = _pair(tmp_path)
    gpu.step([], 100, learning=True, stimulation=([0], 20), lamina_bias=0)
    gpu.step([], 100, learning=True, stimulation=([2], 12), lamina_bias=0)
    assert gpu.weight[0] != 20.
    gpu.reset()
    assert gpu.weight[0] == 20.
    device_weight = float(gpu._csr_weight[int(gpu.circuit['edges'][0])].get())
    assert device_weight == 20.
    np.testing.assert_array_equal(gpu.v.get(), gpu.rest)


def test_determinism_repeated_learning_run_from_identical_state(tmp_path):
    p = _graph(tmp_path)
    c = _circuit()
    mask = np.array([0, 0, 1, 0])
    results = []
    for _ in range(2):
        gpu = GPUMemoryBrain(p, eta=.001, circuit=c, modulation_mask=mask)
        total_spikes = 0
        for stimulus in [([0], 20), ([2], 20), ([0, 2], 20)]:
            counts, _ = gpu.step([], 100, learning=True, stimulation=stimulus, lamina_bias=0)
            total_spikes += int(counts.sum())
        results.append((total_spikes, float(gpu.weight[0]), float(gpu.memory_w[0])))
    assert results[0] == results[1]
