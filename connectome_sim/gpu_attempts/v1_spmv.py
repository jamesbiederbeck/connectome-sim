"""GPU kernel attempt #1 (historical, not used): cuSPARSE SpMV delivery.

Preserved as-is from the point it was replaced in doom/gpu.py, for reference.
Not imported by any test or by doom/gpu.py. See docs/doom-gpu-kernel-review.md
for why this was replaced: profiling found ~2.2ms/substep here, initially
misdiagnosed as cuSPARSE per-call overhead -- it was actually close to
bandwidth-bound (~94 GB/s of this card's ~192 GB/s peak), which a bandwidth
roofline (25.58M edges read every 0.1ms substep) explains directly. Attempt
#2 (v2_thread_per_row.py) tried to remove the assumed overhead and was
slower still; the warp-per-row design that replaced both is the current
doom/gpu.py.
"""
import math
import time
import numpy as np
from connectome_sim.engine import Brain
from connectome_sim.gpu import _configure_cuda_env


class GPUBrainSpMV(Brain):
    def __init__(self, path, dt=.1):
        super().__init__(path, dt)
        try:
            _configure_cuda_env()
            import cupy as cp
            import cupyx.scipy.sparse as cs
            import scipy.sparse as sp
        except ImportError as e:
            raise RuntimeError(
                'The GPU backend requires cupy. Install doom/requirements-gpu.txt '
                'into this environment (and its documented CUDA_PATH/LD_LIBRARY_PATH) '
                'before using doom.gpu_attempts.v1_spmv.GPUBrainSpMV.') from e
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('No CUDA device visible to cupy.')
        self._cp = cp
        n = self.n
        # CSR as produced by doom/prepare.py is pre-major (row i = edges from
        # neuron i); transpose once here to post-major so `weight_t.dot(x)`
        # with x indexed by pre-synaptic neuron returns a post-indexed result.
        pre_major = sp.csr_matrix((self.weight, self.post, self.ptr), shape=(n, n))
        self._weight_t = cs.csr_matrix(pre_major.T.tocsr())
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
        cp = self._cp
        slots = self._queue.shape[0]
        slot = self.cursor % slots
        future = (self.cursor + self._delay_steps) % slots
        self.refractory = cp.maximum(self.refractory - 1, 0)
        not_refractory = self.refractory == 0
        v_new = -52 + (self.v + 52) * self._av + self.drive * (1 - self._av) + self.g * self._coupling
        self.v = cp.where(not_refractory, v_new, self.v)
        self.g = cp.where(not_refractory, self.g * self._ag, self.g)
        spiked = not_refractory & (self.v > -45)
        self._counts += spiked.astype(cp.int32)
        self._queue[future] |= spiked
        # Delivery from spikes scheduled 1.8ms ago; always run the SpMV
        # (rather than branching on "any spikes queued") so every substep
        # stays a fixed sequence of async device calls with no host sync.
        delivered = self._weight_t.dot(self._queue[slot].astype(cp.float32))
        self.g = self.g + delivered * not_refractory
        self._queue[slot] = False
        # Reset happens on the same substep as spiking (only downstream
        # delivery is delayed) -- see doom/engine.py's identical schedule.
        self.v = cp.where(spiked, -52, self.v)
        self.g = cp.where(spiked, 0, self.g)
        self.refractory = cp.where(spiked, self._rfc_steps, self.refractory)
        self.cursor += 1
