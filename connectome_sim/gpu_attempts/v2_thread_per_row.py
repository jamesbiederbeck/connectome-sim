"""GPU kernel attempt #2 (historical, not used): one CUDA thread per row.

Preserved as-is from the point it was replaced in doom/gpu.py, for reference.
Not imported by any test or by doom/gpu.py. Built to remove what was (at the
time, incorrectly) assumed to be cuSPARSE per-call overhead in attempt #1
(v1_spmv.py); measured slower still (~3.9ms/substep vs ~2.2ms), because this
graph's in-degree is extremely skewed (median 112, max 11,203) and a CUDA
warp only runs as fast as its slowest thread -- one hub neuron sharing a warp
with 31 median-degree neurons stalls the whole warp ~100x longer than
needed. The warp-per-row design that fixed this is the current doom/gpu.py;
see docs/doom-gpu-kernel-review.md for the full measurement writeup.
"""
import math
import time
import numpy as np
from connectome_sim.engine import Brain
from connectome_sim.gpu import _configure_cuda_env

_FUSED_SUBSTEP_SOURCE = r'''
extern "C" __global__
void lif_substep_fused(
    const long long* __restrict__ ptr, const int* __restrict__ pre_idx,
    const float* __restrict__ weight, float* v, float* g, int* refractory,
    const float* __restrict__ drive, const bool* __restrict__ delivering,
    bool* __restrict__ future_queue, int* counts,
    float av, float ag, float coupling, int rfc_steps, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
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
    float delta = 0.f;
    for (long long e = ptr[i]; e < ptr[i + 1]; e++)
        if (delivering[pre_idx[e]]) delta += weight[e];
    if (not_refractory) gi += delta;
    if (spiked) { vi = -52.f; gi = 0.f; refr = rfc_steps; }
    v[i] = vi; g[i] = gi; refractory[i] = refr;
}
'''


class GPUBrainThreadPerRow(Brain):
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
                'before using doom.gpu_attempts.v2_thread_per_row.GPUBrainThreadPerRow.') from e
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
