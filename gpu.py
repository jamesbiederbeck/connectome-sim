"""CuPy GPU implementation of the fixed-step LIF model in doom/engine.py.

Event-driven design, matching the CPU kernels' (kernel.cpp, doom/engine.py)
behavior: edges are only touched for neurons that actually spiked, not for
every neuron every substep. This replaces three earlier dense-every-substep
designs (SpMV, thread-per-row, warp-per-row -- preserved in doom/gpu_attempts/
for reference) that were bandwidth-bound: reading all 25,582,938 edges every
dt=0.1ms substep regardless of how many neurons actually spiked (typically
~62 of 166,700) capped them at ~2-3.3 tics/sec on this hardware, no matter
how well-written the kernel was -- see docs/doom-gpu-kernel-review.md for
that roofline math and the measured results of this design.

Two CUDA kernels run per substep, launched back to back on the default
stream (ordering matters and is guaranteed by same-stream sequencing):

1. `lif_decay_spike_reset` -- dense elementwise pass over all n neurons:
   decay, threshold, immediate reset on spike (matching the reference's
   same-tick reset / delayed-delivery schedule), and on spike, atomically
   append the neuron's index into the *future* delay slot's index list
   (an `atomicAdd` on that slot's counter, then a write of the index).
   This part stays dense (not lazy/active-list, unlike kernel.cpp) because
   it's uniform, branch-free SIMD work and already cheap (~0.6ms/substep
   measured in the dense designs) -- the expensive part was always the
   all-edges gather, not this.
2. `lif_deliver_scatter` -- event-driven: reads the *current* slot's spike
   count directly from device memory (no host sync) and grid-strides one
   warp per queued spiking neuron over the graph's native **pre-major**
   CSR (ptr/post/weight exactly as prepare.py produces them -- no
   transpose needed, unlike the dense designs' post-major gather layout),
   scattering `atomicAdd(&g[post[e]], weight[e])` into each destination,
   guarded by that destination's refractory state. Runs strictly after
   kernel 1 (same-stream ordering) since it depends on kernel 1's freshly
   updated refractory array, exactly as doom/engine.py's advance() does.

The delay ring buffer is an index list (`(slots, n)` int32, worst case one
slot holding all n neurons, ~12.7MB -- trivial on a 6GB card) plus a
`(slots,)` int32 counter array, mirroring doom/engine.py's queue/queue_count
exactly, rather than the dense designs' `(slots, n)` boolean array that
required scanning n entries whether or not anything was queued.

Known, accepted cost: `atomicAdd` on float32 `g` makes accumulation order
nondeterministic run-to-run on the GPU (not just different from the CPU, as
every prior design already was). doom/validate_gpu.py measures this
directly (a self-vs-self repeat run) rather than assuming it away -- see the
doc for the measured numbers.

Requires doom/requirements-gpu.txt and a CUDA GPU; not used by default.
"""
import hashlib
import math
import os
import time
from pathlib import Path
import numpy as np
from connectome_sim.engine import Brain
from connectome_sim.photoreceptor import adapted_drive


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


# Dense elementwise pass, one thread per neuron: decay/threshold/reset, same
# math as the dense designs' fused kernel minus the gather. On spike, the
# reset is immediate (matching the reference's same-tick reset / delayed-
# delivery schedule) and the neuron's index is atomically appended to the
# future delay slot's index list instead of setting a bit in a dense array.
_DECAY_SPIKE_RESET_SOURCE = r'''
extern "C" __global__
void lif_decay_spike_reset(
    float* v, float* g, int* refractory, const float* __restrict__ drive,
    int* __restrict__ future_queue, int* future_count, int* counts,
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
    if (spiked) {
        counts[i] += 1;
        int slot = atomicAdd(future_count, 1);
        future_queue[slot] = i;
        vi = -52.f; gi = 0.f; refr = rfc_steps;
    }
    v[i] = vi; g[i] = gi; refractory[i] = refr;
}
'''

# Event-driven scatter delivery: one warp per queued spiking neuron, lanes
# splitting that neuron's out-edges (pre-major CSR, stride 32) and scattering
# directly into each destination's conductance with atomicAdd, guarded by the
# destination's (already-updated-this-substep) refractory state. The spike
# count is read from device memory at kernel start (no host sync needed) and
# the grid-stride loop covers exactly that many queued neurons, however many
# there are -- unlike the dense designs, work here scales with actual spike
# activity (~62/substep), not with n.
_DELIVER_SCATTER_SOURCE = r'''
extern "C" __global__
void lif_deliver_scatter(
    const long long* __restrict__ ptr, const int* __restrict__ post,
    const float* __restrict__ weight, const int* __restrict__ queue,
    const int* __restrict__ queue_count, const int* __restrict__ refractory,
    float* g)
{
    int count = *queue_count;
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;
    int global_warp = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    int num_warps = gridDim.x * warps_per_block;
    for (int k = global_warp; k < count; k += num_warps) {
        int i = queue[k];
        for (long long e = ptr[i] + lane; e < ptr[i + 1]; e += 32) {
            int j = post[e];
            if (refractory[j] == 0) atomicAdd(&g[j], weight[e]);
        }
    }
}
'''


# Analogous to doom/native.py's BUILD sidecar: an auditable record of exactly
# which kernel code produced a run's output. There is no separate compiled
# binary to hash here (cupy.RawKernel JIT-compiles this source via NVRTC at
# construction time, not build time), so kernel_source_sha256 covers the two
# CUDA source strings directly instead of a build artifact.
GPU_BUILD = {
    'model_revision': 'lif-gpu-event-driven-v1',
    'kernel_source_sha256': hashlib.sha256(
        (_DECAY_SPIKE_RESET_SOURCE + _DELIVER_SCATTER_SOURCE).encode()).hexdigest(),
    'binary_sha256': None,
    'compile_flags': ['JIT-compiled via cupy.RawKernel/NVRTC at GPUBrain construction time; no static binary artifact'],
}


class GPUBrain(Brain):
    def __init__(self, path, dt=.1, tau_m=20., tau_s=5.):
        super().__init__(path, dt, tau_m, tau_s)
        if not self.reference_dynamics:
            raise ValueError(
                'This backend compiles the 20 ms / 5 ms time constants into its '
                'kernel and cannot honour tau_m/tau_s. Use the numba Brain for '
                'non-reference dynamics, or extend the kernel signature.')
        try:
            _configure_cuda_env()
            import cupy as cp
        except ImportError as e:
            raise RuntimeError(
                'The GPU backend requires cupy. Install doom/requirements-gpu.txt '
                'into this environment (and its documented CUDA_PATH/LD_LIBRARY_PATH) '
                'before using connectome_sim.gpu.GPUBrain.') from e
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('No CUDA device visible to cupy.')
        self._cp = cp
        n = self.n
        # Native pre-major layout (prepare.py's ptr/post/weight): row i
        # is neuron i's out-edges, exactly what the scatter kernel needs --
        # no transpose, unlike the dense designs' post-major gather layout.
        self._csr_ptr = cp.asarray(self.ptr, dtype=cp.int64)
        self._csr_post = cp.asarray(self.post, dtype=cp.int32)
        self._csr_weight = cp.asarray(self.weight, dtype=cp.float32)
        self._decay_kernel = cp.RawKernel(_DECAY_SPIKE_RESET_SOURCE, 'lif_decay_spike_reset')
        self._deliver_kernel = cp.RawKernel(_DELIVER_SCATTER_SOURCE, 'lif_deliver_scatter')
        self._decay_block = 256
        self._decay_grid = (n + self._decay_block - 1) // self._decay_block
        # Delivery grid is sized in warps, not by n -- it grid-strides over
        # however many neurons are actually queued each substep (read from
        # device memory at kernel start), so this many resident warps is
        # simply "enough to keep the GPU busy," not "one per row."
        self._deliver_block = 256
        self._deliver_grid = max(1, (n // 32 + self._deliver_block - 1) // self._deliver_block)
        self.v = cp.asarray(self.v)
        self.g = cp.asarray(self.g)
        self.refractory = cp.asarray(self.refractory.astype(np.int32))
        self.drive = cp.asarray(self.drive)
        self._delay_steps = int(round(1.8 / dt))
        self._rfc_steps = int(round(2.2 / dt))
        slots = self._delay_steps + 1
        # Index-list ring buffer (mirrors doom/engine.py's queue/queue_count):
        # worst case one slot holds all n neurons, ~12.7MB total, trivial on
        # a 6GB card. Zeroed once here; each slot's counter is zeroed again
        # by _substep right after that slot is consumed, ready for reuse
        # `slots` steps later.
        self._queue = cp.zeros((slots, n), dtype=cp.int32)
        self._queue_count = cp.zeros(slots, dtype=cp.int32)
        self._counts = cp.zeros(n, dtype=cp.int32)
        self._av = math.exp(-dt / 20)
        self._ag = math.exp(-dt / 5)
        self._coupling = (self._av - self._ag) / 3
        self._snapshot_state()

    # __init__ replaced v/g/drive/refractory with cupy arrays and moved
    # queue/queue_count/counts onto private GPU-resident buffers
    # (_queue/_queue_count/_counts) -- the inherited numpy queue/queue_count/
    # counts fields still exist but are dead, unused weight, so they're
    # dropped from STATE_FIELDS rather than reset for no reason. No reset()
    # override needed: Brain.reset()'s copy-on-write loop works identically
    # on cupy arrays (.copy() is defined the same way), given the right field
    # names.
    STATE_FIELDS=('v','g','drive','refractory','_queue','_queue_count','_counts',
                  'luminance','retinal_adaptation','active','active_flag','nactive')

    def step(self, luminance, duration_ms, sugar=False, lamina_bias=12.0, stimulation=None):
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
        drive_host[self.retina] = adapted_drive(self.luminance, self.retinal_adaptation,
                                                steps * self.dt,
                                                tau_ms=self.retinal_adaptation_ms)
        if sugar: drive_host[self.sugar] = 30
        if stimulation is not None:
            pulses = stimulation if isinstance(stimulation, list) else [stimulation]
            for indices, current in pulses:
                ix = np.asarray(indices, dtype=np.int32)
                amplitude = np.asarray(current, dtype=np.float32)
                if ix.ndim != 1 or np.any(ix < 0) or np.any(ix >= self.n) or not np.isfinite(amplitude).all() or amplitude.shape not in [(), ix.shape]:
                    raise ValueError('Invalid external stimulation')
                drive_host[ix] += amplitude
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
        # Kernel 1 must complete before kernel 2 launches (same-stream
        # ordering guarantees this): kernel 2 reads refractory as kernel 1
        # just updated it, and reads future_queue/future_count as kernel 1
        # is about to fill them for a *later* substep, never this one's
        # delivery -- delivery here consumes `slot`, filled `delay_steps`
        # substeps ago, not `future`.
        self._decay_kernel((self._decay_grid,), (self._decay_block,), (
            self.v, self.g, self.refractory, self.drive,
            self._queue[future], self._queue_count[future:future + 1], self._counts,
            cp.float32(self._av), cp.float32(self._ag),
            cp.float32(self._coupling), cp.int32(self._rfc_steps), cp.int32(self.n)))
        self._deliver_kernel((self._deliver_grid,), (self._deliver_block,), (
            self._csr_ptr, self._csr_post, self._csr_weight,
            self._queue[slot], self._queue_count[slot:slot + 1], self.refractory, self.g))
        # Consumed slot's counter must be back at zero before this slot is
        # reused as `future` `slots` substeps from now.
        self._queue_count[slot] = 0
        self.cursor += 1
