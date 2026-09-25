"""GPU-accelerated port of MemoryBrain's per-substep learning kernel.

Extends `connectome_sim.gpu.GPUBrain` (the event-driven CuPy port of the
plain LIF kernel, `kernel.cpp`'s `neural_advance`) with the learning-specific
additions `physiology/kernel.cpp`'s `memory_advance` makes on top of that
same baseline integration: a per-cell resting potential (`rest[i]`,
replacing the hardcoded -52 mV), an adaptation current (`adaptation[i]`,
exponential decay every substep, `adapt_mask[i]`-gated jump on spike), and a
`modulation_mask[i]` guard in the delivery loop.

Scope -- read this before assuming feature parity with `memory_advance`'s
full C signature:

1. **Ported, in the decay/reset kernel** (`_MEMORY_DECAY_SPIKE_RESET_SOURCE`):
   `rest[i]`, `adaptation[i]`'s exponential decay and the `adapt_mask[i]`-
   gated jump on spike, and the subthreshold voltage term adaptation drives.
   This is elementwise, one-thread-per-neuron work, so it is exactly
   `gpu.py`'s `_DECAY_SPIKE_RESET_SOURCE` plus this delta -- no new kernel
   shape needed. The per-thread math was derived by transcribing
   `memory_advance`'s `evolve()` lambda for the dense d=1-substep case (every
   neuron is visited every substep here, unlike the CPU's lazy active-list
   skip): whether a cell is "refractory" or "integrating" this substep,
   `adaptation[i]` decays by the same one-substep factor `aa =
   exp(-dt/adaptation_tau)` either way (`memory_advance` reaches that via two
   different code paths -- `adaptation[i] *= aa[skip]` on the refractory
   branch, `adaptation[i] *= c` on the integration branch -- that collapse to
   the same constant at d=1); the `v -= adaptation*coeff*(c-a)` subtraction
   only applies on the integration branch, using the pre-decay adaptation
   value, exactly as `memory_advance` line order requires (subtract, then
   decay).
2. **Ported, in the delivery kernel** (`_MEMORY_DELIVER_SCATTER_SOURCE`):
   `modulation_mask[i]` skips a spiking modulation cell's entire edge loop,
   so it delivers no normal transmitter to its targets -- the
   correctness-critical part of `memory_advance`'s modulation branch for a
   port that does not implement point 3 below. Every other queued spiker is
   delivered exactly as `gpu.py`'s plain scatter kernel already does.
3. **Deliberately NOT ported**: modulator delivery to targets
   (`modulation[]`/`modulation_last[]`), eligibility trace bookkeeping
   (`eligibility[]`/`eligibility_last[]`), and the in-kernel DAN-gated weight
   decay (`dan_gain`/`baseline_weight`/`eta`/`floor_fraction`, the
   `weight[edge] = candidate>lower?candidate:lower` update).
   `memory_advance` only runs that in-kernel decay path when
   `learning_enabled=1`, and `physiology/brain.py`'s `MemoryBrain.step()` --
   the CPU wrapper this class mirrors -- always calls its native `advance()`
   with `learning=False`: on the CPU side too, that in-kernel path is
   currently vestigial dead code for this run mode. The actual weight update
   happens separately in Python, in `rule.py`'s `advance()`, over the
   circuit's few thousand plastic edges per <=100-tick/10ms rate bin -- cheap
   relative to the 166,700-neuron/25.6M-edge kernel call, and explicitly out
   of scope for this port (a future GPU caller of `rule.py` is a mechanical
   numpy->cupy swap, not attempted here). `GPUMemoryBrain.step()` below calls
   `rule.advance()` exactly the way `MemoryBrain.step()` does, unchanged,
   layered on top of this GPU-accelerated substep kernel underneath.

   **Consequence for a caller**: a GPU learning run drops modulator
   delivery/eligibility/in-kernel weight-decay relative to `memory_advance`'s
   full signature. This only diverges from the CPU path's *observable*
   behavior if a future caller sets `learning_enabled=1` on the CPU side too
   (nothing here does). Say so plainly rather than silently omitting it.

4. **Plastic-weight upload**: after `rule.advance()` writes
   `self.weight[circuit['edges']] = baseline_plastic*(1+memory_w)` (a host
   numpy write, unchanged from `MemoryBrain.step()`), the *device* copy the
   kernel actually reads (`self._csr_weight`) must be updated too, or the GPU
   substep loop never sees a learned weight and silently behaves like a
   frozen-weight run. This is done by scattering just the plastic edges'
   values (`self._csr_weight[self._plastic_edges_gpu] = ...`), not by
   re-uploading the full 25.6M-edge array every bin.

See `docs/gpu-learning-kernel-review.md` for the Tier 1/Tier 2/benchmark/
determinism validation this module was held to.
"""
import hashlib
import math
import time
import numpy as np
from connectome_sim.gpu import GPUBrain
from connectome_sim.photoreceptor import adapted_drive
from connectome_sim.physiology.common import GRAPH, ROOT, digest
from connectome_sim.physiology.circuit import identify

MODEL = 'adaptive-centered-v6-gpu'

# Dense elementwise pass, one thread per neuron -- gpu.py's
# _DECAY_SPIKE_RESET_SOURCE plus exactly the learning delta described in
# point 1 of the module docstring above. `rest`/`adaptation`/`adapt_mask` are
# extra __restrict__ input/state arrays; nothing about the one-thread-per-
# neuron structure or the delay-queue enqueue changes.
_MEMORY_DECAY_SPIKE_RESET_SOURCE = r'''
extern "C" __global__
void lif_memory_decay_spike_reset(
    float* v, float* g, int* refractory, const float* __restrict__ drive,
    int* __restrict__ future_queue, int* future_count, int* counts,
    float* adaptation, const float* __restrict__ rest,
    const unsigned char* __restrict__ adapt_mask,
    float av, float ag, float coupling, float aa, float adapt_coeff,
    float adaptation_jump, int rfc_steps, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    int refr = refractory[i]; refr = refr > 0 ? refr - 1 : 0;
    bool not_refractory = (refr == 0);
    float vi = v[i], gi = g[i], ai = adaptation[i], ri = rest[i];
    if (not_refractory) {
        float v_new = ri + (vi - ri) * av + drive[i] * (1.f - av) + gi * coupling;
        gi = gi * ag;
        // Subtract using the PRE-decay adaptation value, matching
        // memory_advance's evolve(): the subtraction (kernel.cpp:33) runs
        // before adaptation[i] *= c on the same line.
        vi = v_new - ai * adapt_coeff * (aa - av);
    }
    // Decays by the same one-substep factor whichever branch ran above --
    // see module docstring point 1 for why this is exact at d=1.
    ai *= aa;
    bool spiked = not_refractory && (vi > -45.f);
    if (spiked) {
        counts[i] += 1;
        int slot = atomicAdd(future_count, 1);
        future_queue[slot] = i;
        vi = ri; gi = 0.f; refr = rfc_steps;
        if (adapt_mask[i]) ai += adaptation_jump;
    }
    v[i] = vi; g[i] = gi; refractory[i] = refr; adaptation[i] = ai;
}
'''

# gpu.py's _DELIVER_SCATTER_SOURCE plus the modulation_mask guard described
# in point 2 of the module docstring: a modulation cell's spike still enters
# the delay queue and still decays/resets normally in the kernel above (it is
# a real spike), but its edge loop here is skipped entirely, so it delivers
# no normal transmitter to any target -- see point 3 for what this drops.
_MEMORY_DELIVER_SCATTER_SOURCE = r'''
extern "C" __global__
void lif_memory_deliver_scatter(
    const long long* __restrict__ ptr, const int* __restrict__ post,
    const float* __restrict__ weight, const int* __restrict__ queue,
    const int* __restrict__ queue_count, const int* __restrict__ refractory,
    const unsigned char* __restrict__ modulation_mask, float* g)
{
    int count = *queue_count;
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;
    int global_warp = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    int num_warps = gridDim.x * warps_per_block;
    for (int k = global_warp; k < count; k += num_warps) {
        int i = queue[k];
        // Correctness-critical drop, not a silent omission: see module
        // docstring points 2-3 for exactly what a modulation cell's spike
        // no longer does in this port (deliver transmitter, update
        // modulation[]/eligibility[], gate in-kernel weight decay).
        if (modulation_mask[i]) continue;
        for (long long e = ptr[i] + lane; e < ptr[i + 1]; e += 32) {
            int j = post[e];
            if (refractory[j] == 0) atomicAdd(&g[j], weight[e]);
        }
    }
}
'''

GPU_MEMORY_BUILD = {
    'model_revision': 'lif-memory-gpu-event-driven-v1',
    'kernel_source_sha256': hashlib.sha256(
        (_MEMORY_DECAY_SPIKE_RESET_SOURCE + _MEMORY_DELIVER_SCATTER_SOURCE).encode()).hexdigest(),
    'binary_sha256': None,
    'compile_flags': ['JIT-compiled via cupy.RawKernel/NVRTC at GPUMemoryBrain construction time; no static binary artifact'],
    'scope': ('Ports memory_advance\'s per-neuron rest/adaptation math and the '
              'modulation_mask delivery guard only. Drops modulator delivery, '
              'eligibility bookkeeping and in-kernel DAN-gated weight decay '
              '(all only reachable with learning_enabled=1, which '
              'MemoryBrain.step() never sets); rule.py stays on CPU unmodified. '
              'See physiology/gpu_brain.py module docstring.'),
}


class GPUMemoryBrain(GPUBrain):
    """GPU sibling of `MemoryBrain`, not a flag on `GPUBrain`/`MemoryBrain`.

    Needs its own device-resident `rest`/`adaptation`/`adapt_mask`/
    `modulation_mask` state and its own `step()`/`_substep()`/`reset()`, the
    same shape `GPUBrain.reset()` already needed for its own `_queue`/
    `_counts`/`_queue_count` buffers. Exposes `step`, `reset`, `memory()` and
    the circuit/eta/rest/adaptation setup from `MemoryBrain.__init__` so
    `rule.py`'s `advance()` can be called against its returned counts exactly
    the way `MemoryBrain.step()` already does -- the outer rate-binning +
    `rule.advance()` + weight-write loop is unchanged CPU logic layered on
    top of a GPU-accelerated per-substep kernel underneath, not
    reimplemented.
    """

    def __init__(self, path=GRAPH, *, dt=.1, eta=.001, circuit=None, circuit_spec=None,
                 modulation_mask=None, tonic=None, dan_baseline_hz=None, kc_rest=-60.,
                 adaptation_jump=8., adaptation_tau=200., adaptation_mask=None):
        super().__init__(path, dt=dt)
        cp = self._cp
        self.circuit = identify(self, spec=circuit_spec) if circuit is None else circuit
        if not math.isfinite(kc_rest) or not -80 <= kc_rest <= -45:
            raise ValueError('Invalid KC resting potential')
        self.rest = np.full(self.n, -52., dtype=np.float32)
        self.rest[self.circuit['kc']] = kc_rest
        # GPUBrain.__init__ already set self.v to a flat -52 cupy array;
        # override with the per-cell resting potential, matching
        # MemoryBrain.__init__'s self.v[:] = self.rest.
        self.v = cp.asarray(self.rest)
        if (not math.isfinite(adaptation_jump) or adaptation_jump < 0
                or not math.isfinite(adaptation_tau) or adaptation_tau <= 20):
            raise ValueError('Invalid adaptation parameters')
        self.adaptation_jump = float(adaptation_jump)
        self.adaptation_tau = float(adaptation_tau)
        self.adaptation_mask = (np.asarray(self.circuit['kc_mask'], dtype=np.uint8).copy()
                                 if adaptation_mask is None
                                 else np.asarray(adaptation_mask, dtype=np.uint8).copy())
        if self.adaptation_mask.shape != (self.n,) or np.any(self.adaptation_mask > 1):
            raise ValueError('Invalid adaptation mask')
        if modulation_mask is None:
            import pyarrow.feather as feather
            neurons = feather.read_table(
                ROOT / 'connectome_data/malecns_v1/normalized/neurons.feather'
            ).to_pandas().set_index('source_id').loc[self.ids]
            modulation_mask = neurons.neurotransmitter.isin(
                ['dopamine', 'octopamine', 'serotonin']).to_numpy(dtype=np.uint8)
        self.modulation_mask = np.asarray(modulation_mask, dtype=np.uint8).copy()
        if self.modulation_mask.shape != (self.n,) or np.any(self.modulation_mask > 1):
            raise ValueError('Invalid modulation mask')
        self.eta = float(eta)
        if not math.isfinite(self.eta) or self.eta < 0:
            raise ValueError('Finite nonnegative eta required')
        self.baseline_plastic = self.weight[self.circuit['edges']].copy()
        self.initial_weight_sha256 = digest(self.weight)
        self.rate_kc = np.zeros(len(self.circuit['edges']), dtype=np.float64)
        self.rate_dan = np.zeros(len(self.circuit['dan']), dtype=np.float64)
        self.memory_u = np.zeros_like(self.rate_kc)
        self.memory_w = np.zeros_like(self.rate_kc)
        self.tonic = np.zeros(self.n, dtype=np.float32) if tonic is None else np.asarray(tonic, dtype=np.float32).copy()
        self.dan_baseline_hz = (np.zeros(len(self.circuit['dan']), dtype=np.float64) if dan_baseline_hz is None
                                 else np.asarray(dan_baseline_hz, dtype=np.float64).copy())
        if self.tonic.shape != (self.n,) or not np.isfinite(self.tonic).all():
            raise ValueError('Invalid tonic current')
        if self.dan_baseline_hz.shape != (len(self.rate_dan),) or not np.isfinite(self.dan_baseline_hz).all():
            raise ValueError('Invalid DAN baseline')
        self.weights_frozen = False

        # Device-resident learning state.
        self._rest_gpu = cp.asarray(self.rest)
        self._adaptation = cp.zeros(self.n, dtype=cp.float32)
        self._adapt_mask_gpu = cp.asarray(self.adaptation_mask)
        self._modulation_mask_gpu = cp.asarray(self.modulation_mask)
        # Scatter target for the outer loop's weight write -- see module
        # docstring point 4. int64 to match self._csr_weight's edge indexing.
        self._plastic_edges_gpu = cp.asarray(self.circuit['edges'].astype(np.int64))
        self._aa = math.exp(-dt / self.adaptation_tau)
        self._adapt_coeff = self.adaptation_tau / (self.adaptation_tau - 20.)

        self._memory_decay_kernel = cp.RawKernel(_MEMORY_DECAY_SPIKE_RESET_SOURCE, 'lif_memory_decay_spike_reset')
        self._memory_deliver_kernel = cp.RawKernel(_MEMORY_DELIVER_SCATTER_SOURCE, 'lif_memory_deliver_scatter')
        self._snapshot_state()

    # GPUBrain.STATE_FIELDS plus this class's own device-resident learning
    # state. `v` is already in GPUBrain.STATE_FIELDS and is snapshotted here
    # holding the per-cell resting potential (`self.v = cp.asarray(self.rest)`
    # above overwrote GPUBrain's flat -52 default before this line runs), so
    # the generic copy-on-write restore needs no special case for it -- unlike
    # the old reset() below, which had to remember `self.v[:] = self._rest_gpu`
    # by hand. `_adaptation` is this class's device-resident equivalent of
    # `MemoryBrain.adaptation` (a different name because it lives on the GPU,
    # not because it means something different).
    STATE_FIELDS=GPUBrain.STATE_FIELDS+('_adaptation','rate_kc','rate_dan','memory_u','memory_w')

    def reset(self, keep_memory=False):
        """Copy-on-write reset (see Brain._snapshot_state) plus the plastic
        KC->MBON11 weight subset on both the host array and its device
        mirror (`_csr_weight`, which the GPU kernel actually reads -- see
        the module docstring's point 4), and `keep_memory`, same meaning as
        `MemoryBrain.reset`.
        """
        skip={'memory_u','memory_w'} if keep_memory else set()
        for name,snapshot in self._initial_state.items():
            if name in skip:continue
            setattr(self,name,snapshot.copy())
        for name,value in self.STATE_SCALARS.items():
            setattr(self,name,value)
        if not keep_memory:
            self.weight[self.circuit['edges']]=self.baseline_plastic
            self._csr_weight[self._plastic_edges_gpu]=self._cp.asarray(self.baseline_plastic)

    def _substep(self):
        cp = self._cp
        slots = self._queue.shape[0]
        slot = self.cursor % slots
        future = (self.cursor + self._delay_steps) % slots
        self._memory_decay_kernel((self._decay_grid,), (self._decay_block,), (
            self.v, self.g, self.refractory, self.drive,
            self._queue[future], self._queue_count[future:future + 1], self._counts,
            self._adaptation, self._rest_gpu, self._adapt_mask_gpu,
            cp.float32(self._av), cp.float32(self._ag), cp.float32(self._coupling),
            cp.float32(self._aa), cp.float32(self._adapt_coeff), cp.float32(self.adaptation_jump),
            cp.int32(self._rfc_steps), cp.int32(self.n)))
        self._memory_deliver_kernel((self._deliver_grid,), (self._deliver_block,), (
            self._csr_ptr, self._csr_post, self._csr_weight,
            self._queue[slot], self._queue_count[slot:slot + 1], self.refractory,
            self._modulation_mask_gpu, self.g))
        self._queue_count[slot] = 0
        self.cursor += 1

    def _neural_step(self, luminance, duration_ms, *, learning=False, stimulation=None, lamina_bias=12.):
        # `learning` is accepted only for interface parity with
        # MemoryBrain._neural_step; step() below always calls this with
        # learning=False, exactly as MemoryBrain.step() does -- see module
        # docstring point 3.
        cp = self._cp
        light = np.asarray(luminance)
        if light.shape != (len(self.retina),) or not np.isfinite(light).all():
            raise ValueError('Invalid retinal input')
        steps = round(duration_ms / self.dt)
        if not math.isfinite(duration_ms) or steps < 1 or not math.isfinite(lamina_bias):
            raise ValueError('Invalid interval/current')
        self.luminance += (1 - math.exp(-steps * self.dt / 10)) * (np.clip(light, 0, 1) - self.luminance)
        drive_host = np.zeros(self.n, dtype=np.float32)
        drive_host[self.lamina] = lamina_bias
        drive_host[self.retina] = adapted_drive(self.luminance, self.retinal_adaptation,
                                                 steps * self.dt, tau_ms=self.retinal_adaptation_ms)
        drive_host += self.tonic
        if stimulation is not None:
            pulses = stimulation if isinstance(stimulation, list) else [stimulation]
            for indices, current in pulses:
                ix = np.asarray(indices, dtype=np.int32)
                amplitude = np.asarray(current, dtype=np.float32)
                if (ix.ndim != 1 or np.any(ix < 0) or np.any(ix >= self.n)
                        or not np.isfinite(amplitude).all() or amplitude.shape not in [(), ix.shape]):
                    raise ValueError('Invalid external stimulation')
                drive_host[ix] += amplitude
        self.drive = cp.asarray(drive_host)
        self._counts.fill(0)
        start = time.perf_counter()
        for _ in range(steps):
            self._substep()
        counts = self._counts.get()
        elapsed = time.perf_counter() - start
        self.sim_ms = self.cursor * self.dt
        self.total_spikes += int(counts.sum())
        return counts, elapsed

    def step(self, luminance, duration_ms, *, learning=False, stimulation=None, lamina_bias=12.):
        from connectome_sim.physiology.rule import advance
        if not math.isfinite(duration_ms) or duration_ms <= 0:
            raise ValueError('Invalid duration')
        remaining = round(duration_ms / self.dt)
        if remaining < 1:
            raise ValueError('Duration too short')
        cp = self._cp
        total = np.zeros(self.n, dtype=np.int32)
        wall = 0.
        while remaining:
            ticks = min(100, remaining)
            interval = ticks * self.dt
            # The original LTD update is disabled, matching MemoryBrain.step():
            # only the centered rule below writes candidate memory
            # efficacies; all neural integration remains, on GPU here.
            c, t = self._neural_step(luminance, interval, learning=False,
                                      stimulation=stimulation, lamina_bias=lamina_bias)
            seconds = interval / 1000
            advance(self.rate_kc, self.rate_dan, self.memory_u, self.memory_w,
                    c[self.circuit['pre']] / seconds, c[self.circuit['dan']] / seconds - self.dan_baseline_hz,
                    self.circuit['gain'], seconds, self.eta, learning, self.weights_frozen)
            if not self.weights_frozen:
                self.weight[self.circuit['edges']] = self.baseline_plastic * (1 + self.memory_w)
                # See module docstring point 4: scatter only the plastic
                # edges into the device weight array, not a full re-upload.
                self._csr_weight[self._plastic_edges_gpu] = cp.asarray(self.weight[self.circuit['edges']])
            total += c
            wall += t
            remaining -= ticks
        self.counts[:] = total
        return total, wall

    def memory(self):
        w = self.weight[self.circuit['edges']]
        fraction = w / self.baseline_plastic
        return {'plastic_edges': len(w), 'changed_edges': int(np.count_nonzero(w != self.baseline_plastic)),
                'mean_efficacy': float(fraction.mean()), 'minimum_efficacy': float(fraction.min()),
                'sha256': digest(w), 'model': MODEL}
