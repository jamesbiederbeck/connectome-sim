"""CuPy GPU implementation of the fixed-step LIF model in doom/engine.py.

Ports the dense every-substep formulation (not kernel.cpp's lazy active-list
skip, which is a CPU-only optimization that becomes a liability on a GPU).
Synaptic delivery, decay, threshold and reset are fused into one hand-written
CUDA kernel launch per substep (_FUSED_SUBSTEP_SOURCE), against the graph
transposed once at load time to post-major layout.

IMPORTANT, measured limit: the dense formulation reads every one of the
graph's 25,582,938 edges (204.7MB) each dt=0.1ms substep, regardless of how
many neurons actually spiked (typically ~62 of 166,700). That is bandwidth
work, not a kernel-quality problem -- see docs/doom-gpu-kernel-review.md for
the roofline math. Two kernel designs were tried, in order:

1. An SpMV-via-cuSPARSE version measured ~2.2ms/substep (~94 GB/s, ~49% of
   this card's ~192 GB/s peak) -- already close to bandwidth-bound, not
   overhead-bound as first assumed.
2. A hand-written one-thread-per-row replacement was *slower* (~3.9ms/substep,
   ~52 GB/s): this graph's in-degree is extremely skewed (median 112, max
   11,203, 258 neurons above 2,000), and a CUDA warp only runs as fast as its
   slowest thread, so one hub neuron sharing a warp with 31 median-degree
   neurons stalls the whole warp ~100x longer than needed.

The kernel below assigns one *warp* (32 threads) per row instead, splitting
that row's edges across the warp's lanes and combining partial sums with a
shuffle-reduce, with a grid-stride loop over rows so a warp that finishes a
low-degree row immediately picks up another rather than idling. This reaches
~1.9ms/substep, close to the ~1.6ms roofline ceiling for this design at this
card's peak bandwidth -- i.e. this is close to the best any kernel can do
*for this dense-every-substep architecture on this GPU*; going faster would
need an event-driven design that only reads edges from neurons that actually
spiked (this file's approach reads all of them every substep by construction;
see the doc's "if you want to actually beat the CPU" section for what that
would take, not attempted here).

The reference's "no delivery into an already-refractory neuron" guard
depends only on the destination neuron, never the edge, so it is applied as
a plain conditional (lane 0 only, after the reduction) rather than folded
into the gather. Requires doom/requirements-gpu.txt and a CUDA GPU; not
used by default.
"""
import math
import os
import time
from pathlib import Path
import numpy as np
from doom.engine import Brain


# dependency order matters: cublasLt/cublas before cusolver, nvjitlink/nvrtc
# before anything that JITs, cusparse last of the math libraries.
_PRELOAD_ORDER = ['cudart', 'nvjitlink', 'nvrtc.so', 'cublasLt', 'cublas.so',
                  'curand', 'cusparse', 'cusolver.so', 'cusolverMg']


def _configure_cuda_env():
    """Load the pip-installed nvidia-*-cu12 wheels' shared libraries directly.

    doom/requirements-gpu.txt installs cupy-cuda12x==13.3.0, which (at this
    version) does not auto-locate the separately-wheeled CUDA runtime shared
    libraries the way later cupy/cuda-pathfinder integrations do -- it dlopens
    bare names like "libcusparse.so.12" and expects them on the loader's
    search path. glibc's dynamic linker reads LD_LIBRARY_PATH once at process
    start, so setting os.environ here (after the interpreter is already
    running) does not affect it. Instead, preload each library by absolute
    path with RTLD_GLOBAL: once a shared object is resident, a later bare-name
    dlopen for the same soname is satisfied from the process's already-loaded
    libraries without any path search, so no shell export is required.
    """
    import ctypes
    import importlib.util
    spec = importlib.util.find_spec('nvidia')
    if spec is None or not spec.submodule_search_locations: return
    found, cuda_runtime_root = {}, None
    for root in spec.submodule_search_locations:
        root = Path(root)
        if not root.is_dir(): continue
        for child in sorted(root.iterdir()):
            lib = child / 'lib'
            if not lib.is_dir(): continue
            if child.name == 'cuda_runtime': cuda_runtime_root = child
            for so in lib.glob('*.so*'):
                if '.alt.' in so.name: continue
                found.setdefault(so.name, so)
    for key in _PRELOAD_ORDER:
        for name, so in list(found.items()):
            if key in name:
                try: ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
                except OSError: pass
    if cuda_runtime_root is not None:
        os.environ.setdefault('CUDA_PATH', str(cuda_runtime_root))


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


class GPUBrain(Brain):
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
                'before using doom.gpu.GPUBrain.') from e
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('No CUDA device visible to cupy.')
        self._cp = cp
        n = self.n
        # CSR as produced by doom/prepare.py is pre-major (row i = edges from
        # neuron i); transpose once here to post-major so `weight_t.dot(x)`
        # with x indexed by pre-synaptic neuron returns a post-indexed result.
        pre_major = sp.csr_matrix((self.weight, self.post, self.ptr), shape=(n, n))
        weight_t = pre_major.T.tocsr()
        self._csr_ptr = cp.asarray(weight_t.indptr, dtype=cp.int64)
        self._csr_pre = cp.asarray(weight_t.indices, dtype=cp.int32)
        self._csr_weight = cp.asarray(weight_t.data, dtype=cp.float32)
        self._kernel = cp.RawKernel(_FUSED_SUBSTEP_SOURCE, 'lif_substep_fused')
        # block/grid size the total number of resident warps, not "one warp
        # per row" -- the kernel's own grid-stride loop covers all n rows
        # regardless of how many warps are launched; this many happens to
        # keep every row covered without a stride wraparound too.
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
        # future's row is a full overwrite (not accumulated), so it never
        # needs a separate clear -- each of the `slots` rows is written
        # exactly once (as `future`) and read exactly once (as `slot`,
        # `delay_steps` substeps later) per cycle.
        self._kernel((self._grid,), (self._block,), (
            self._csr_ptr, self._csr_pre, self._csr_weight,
            self.v, self.g, self.refractory, self.drive,
            self._queue[slot], self._queue[future], self._counts,
            self._cp.float32(self._av), self._cp.float32(self._ag),
            self._cp.float32(self._coupling), self._cp.int32(self._rfc_steps),
            self._cp.int32(self.n)))
        self.cursor += 1
