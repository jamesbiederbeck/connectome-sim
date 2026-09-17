"""GPU kernel attempt #3 (historical, not used): dense every-substep, warp-per-row gather.

Preserved as-is from the point it was replaced in doom/gpu.py, for reference.
Not imported by any test or by doom/gpu.py. This was the best of the three
dense-every-substep designs (SpMV, thread-per-row, this one) at 2.064
tics/sec / 56% of this card's ~192 GB/s peak bandwidth -- but the
dense-every-substep formulation itself reads all 25,582,938 edges every
0.1ms substep regardless of how many neurons actually spiked (~62/substep),
a ~3.3 tics/sec architectural ceiling on this hardware at any kernel
quality. doom/gpu.py now implements an event-driven design (two kernels:
dense elementwise decay/threshold/reset+enqueue, then a scatter-delivery
kernel that only touches queued spiking neurons' edges), matching the CPU
kernels' (kernel.cpp, doom/engine.py) event-driven behavior instead of this
file's brute-force gather. See docs/doom-gpu-kernel-review.md for the full
measurement writeup and comparison across all four designs.
"""
import math
import os
import time
from pathlib import Path
import numpy as np
from doom.engine import Brain
from doom.gpu import _configure_cuda_env


# One warp (32 threads) per destination-neuron row: lanes split that row's
# edges (stride 32) and combine partial sums with a shuffle-reduce, then lane
# 0 does the per-neuron decay/threshold/reset. Grid-stride over rows so a
# warp that finishes a low-degree row immediately takes another instead of
# idling -- see the module docstring for why (in-degree here spans 100x,
# median 112 to max 11,203, so one-thread-per-row stalls badly on hubs).
_FUSED_SUBSTEP_SOURCE = r'''
extern "C" __global__
void lif_substep_fused(
    const long long* __restrict__ ptr, const int* __restrict__ pre_idx,
    const float* __restrict__ weight, float* v, float* g, int* refractory,
    const float* __restrict__ drive, const bool* __restrict__ delivering,
    bool* __restrict__ future_queue, int* counts,
    float av, float ag, float coupling, int rfc_steps, int n)
{
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;
    int global_warp = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    int num_warps = gridDim.x * warps_per_block;
    for (int i = global_warp; i < n; i += num_warps) {
        float partial = 0.f;
        for (long long e = ptr[i] + lane; e < ptr[i + 1]; e += 32)
            if (delivering[pre_idx[e]]) partial += weight[e];
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_down_sync(0xffffffff, partial, off);
        if (lane != 0) continue;
        int refr = refractory[i]; refr = refr > 0 ? refr - 1 : 0;
        bool not_refractory = (refr == 0);
        float vi = v[i], gi = g[i];
        if (not_refractory) {
            float v_new = -52.f + (vi + 52.f) * av + drive[i] * (1.f - av) + gi * coupling;
            gi = gi * ag;
            vi = v_new;
        }
        bool spiked = not_refractory && (vi > -45.f);
        if (spiked) counts[i] += 1;
        future_queue[i] = spiked;
        if (not_refractory) gi += partial;
        if (spiked) { vi = -52.f; gi = 0.f; refr = rfc_steps; }
        v[i] = vi; g[i] = gi; refractory[i] = refr;
    }
}
'''


class GPUBrainDenseWarpPerRow(Brain):
    def __init__(self, path, dt=.1):
        super().__init__(path, dt)
        try:
            _configure_cuda_env()
            import cupy as cp
            import scipy.sparse as sp
        except ImportError as e:
            raise RuntimeError(
                'The GPU backend requires cupy. Install doom/requirements-gpu.txt '
                'into this environment (and its documented CUDA_PATH/LD_LIBRARY_PATH) '
                'before using doom.gpu_attempts.v3_dense_warp_per_row.GPUBrainDenseWarpPerRow.') from e
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('No CUDA device visible to cupy.')
        self._cp = cp
        n = self.n
        pre_major = sp.csr_matrix((self.weight, self.post, self.ptr), shape=(n, n))
        weight_t = pre_major.T.tocsr()
        self._csr_ptr = cp.asarray(weight_t.indptr, dtype=cp.int64)
        self._csr_pre = cp.asarray(weight_t.indices, dtype=cp.int32)
        self._csr_weight = cp.asarray(weight_t.data, dtype=cp.float32)
        self._kernel = cp.RawKernel(_FUSED_SUBSTEP_SOURCE, 'lif_substep_fused')
        self._block = 256
        self._grid = (n + self._block - 1) // self._block
        self.v = cp.asarray(self.v)
        self.g = cp.asarray(self.g)
        self.refractory = cp.asarray(self.refractory.astype(np.int32))
        self.drive = cp.asarray(self.drive)
        self._delay_steps = int(round(1.8 / dt))
        self._rfc_steps = int(round(2.2 / dt))
        self._queue = cp.zeros((self._delay_steps + 1, n), dtype=cp.bool_)
        self._counts = cp.zeros(n, dtype=cp.int32)
        self._av = math.exp(-dt / 20)
        self._ag = math.exp(-dt / 5)
        self._coupling = (self._av - self._ag) / 3

    def step(self, luminance, duration_ms, sugar=False, lamina_bias=12.0):
        if len(luminance) != len(self.retina) or not np.all(np.isfinite(luminance)):
            raise ValueError('A finite luminance sample is required for every mapped receptor')
        if not math.isfinite(duration_ms) or not math.isfinite(lamina_bias):
            raise ValueError('Finite duration and current required')
        steps = int(round(duration_ms / self.dt))
        if steps < 1: raise ValueError('Duration too short')
        alpha = 1 - math.exp(-steps * self.dt / 10)
        self.luminance += alpha * (np.clip(luminance, 0, 1) - self.luminance)
        drive_host = np.zeros(self.n, dtype=np.float32)
        drive_host[self.lamina] = lamina_bias
        drive_host[self.retina] = 30 * self.luminance / (.02 + self.luminance)
        if sugar: drive_host[self.sugar] = 30
        self.drive = self._cp.asarray(drive_host)
        self._counts.fill(0)
        start = time.perf_counter()
        for _ in range(steps): self._substep()
        counts = self._counts.get()
        wall = time.perf_counter() - start
        self.total_spikes += int(counts.sum())
        self.sim_ms += steps * self.dt
        return counts, wall

    def _substep(self):
        slots = self._queue.shape[0]
        slot = self.cursor % slots
        future = (self.cursor + self._delay_steps) % slots
        self._kernel((self._grid,), (self._block,), (
            self._csr_ptr, self._csr_pre, self._csr_weight,
            self.v, self.g, self.refractory, self.drive,
            self._queue[slot], self._queue[future], self._counts,
            self._cp.float32(self._av), self._cp.float32(self._ag),
            self._cp.float32(self._coupling), self._cp.int32(self._rfc_steps),
            self._cp.int32(self.n)))
        self.cursor += 1
