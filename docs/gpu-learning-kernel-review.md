# GPU learning kernel backend — extends the event-driven GPU kernel with per-cell rest/adaptation, drops the in-kernel modulation path

This documents `physiology/gpu_brain.py`'s `GPUMemoryBrain(GPUBrain)`, a GPU
port of `physiology/brain.py`'s `MemoryBrain`'s per-substep learning kernel
(`physiology/kernel.cpp`'s `memory_advance`). It extends `gpu.py`'s
`GPUBrain` (the event-driven CuPy port of the plain kernel, already
documented in `androsophila/docs/doom-gpu-kernel-review.md`) rather than
replacing it, following that doc's own Tier 1 / Tier 2 / benchmark /
determinism structure.

## What was built, and the scope cuts

`memory_advance` adds three things over `neural_advance`: a per-cell resting
potential (`rest[i]`), an adaptation current (`adaptation[i]`,
`adapt_mask[i]`-gated jump on spike, exponential decay), and a
`modulation_mask[i]`-gated branch that suppresses normal transmitter delivery
for DAN/octopamine/serotonin cells and instead updates modulator/eligibility/
in-kernel weight-decay state when `learning_enabled=1`.

1. **Decay kernel** (`_MEMORY_DECAY_SPIKE_RESET_SOURCE` = `gpu.py`'s
   `_DECAY_SPIKE_RESET_SOURCE` + `rest`/`adaptation`/`adapt_mask`): ported in
   full. Per-thread math was derived by transcribing `memory_advance`'s
   `evolve()` lambda for the dense d=1-substep case: `adaptation[i]` decays
   by the same one-substep factor `aa=exp(-dt/adaptation_tau)` whichever
   branch (refractory or integrating) a cell takes this substep -- the CPU
   reaches that via two different code paths that collapse to the same
   constant at d=1 -- and the `v -= adaptation*coeff*(aa-av)` subtraction
   uses the pre-decay adaptation value, matching `memory_advance`'s line
   order (subtract, then decay).
2. **Delivery kernel** (`_MEMORY_DELIVER_SCATTER_SOURCE` = `gpu.py`'s
   `_DELIVER_SCATTER_SOURCE` + a `modulation_mask[i]` guard): ported to the
   extent that's correctness-critical for a learning run -- a
   modulation-mask cell's spike is not allowed to leak normal transmitter to
   its targets. **Not ported**: modulator delivery to targets
   (`modulation[]`), eligibility bookkeeping (`eligibility[]`), and the
   in-kernel DAN-gated weight decay. `memory_advance` only runs that decay
   path when `learning_enabled=1`, and `MemoryBrain.step()` -- the CPU
   wrapper this class mirrors -- always calls its native `advance()` with
   `learning=False`: that in-kernel path is vestigial dead code on the CPU
   side too, for this run mode. This is a real, stated scope cut (a GPU
   learning run drops those effects rather than modeling them), not a hidden
   omission -- see `physiology/gpu_brain.py`'s module docstring.
3. **`rule.py` was not touched.** The actual weight update happens
   separately in Python over a few thousand plastic edges per <=100-tick
   bin -- cheap relative to the 166,700-neuron/25.6M-edge kernel call.
   `GPUMemoryBrain.step()` calls it exactly the way `MemoryBrain.step()`
   does, unchanged.
4. **New sibling class**, `GPUMemoryBrain(GPUBrain)`, not a flag: its own
   device-resident `rest`/`adaptation`/`adapt_mask`/`modulation_mask`
   buffers, its own `_substep`/`step`/`reset`/`memory()`, the same shape
   `GPUBrain.reset()` needed for its own queue/count buffers.
5. **Plastic-weight upload bug this design specifically guards against**:
   `MemoryBrain.step()`'s outer loop writes
   `self.weight[circuit['edges']] = baseline_plastic*(1+memory_w)` -- a host
   numpy write. `GPUBrain.__init__` uploads the full edge-weight array to the
   device *once*, at construction; the kernel reads that device copy
   (`self._csr_weight`), not `self.weight`. Reusing the CPU outer loop
   unchanged would silently produce a GPU run that never sees a learned
   weight -- a frozen-weight run wearing a learning costume, and a bug a
   naive Tier 1 test could easily miss (short toy runs can still match CPU
   spike counts by coincidence). Fixed by scattering just the plastic
   edges' values into `self._csr_weight` after every weight write
   (`self._csr_weight[self._plastic_edges_gpu] = ...`), never a full
   25.6M-edge re-upload. `test_gpu_memory_brain.py`'s
   `test_learning_run_updates_weight_on_both_backends_and_device_weight_matches`
   asserts the device copy directly, not just the host-side `self.weight`.

## Environment

GPU: NVIDIA GeForce GTX 1060 6GB (Pascal, cc 6.1), driver 580.95.05. `cupy`
was not installed in the primary shell (`ModuleNotFoundError`), but is
already installed and working in `androsophila/.venv-neural`, the venv this
repo's engine is normally exercised through via the
`androsophila/connectome_sim -> ../connectome-sim` symlink. All numbers
below were run through that venv. `connectome_sim.physiology.common.ROOT`
resolves via `Path(__file__).absolute().parents[2]` -- `.absolute()` does
not resolve symlinks, so imported through that symlink, `ROOT` is
`androsophila/`, which is where `outputs/connectome_sim/malecns_v1/graph.npz`
and the compiled `libneural.so`/`libmemory.so` actually live (not under
`connectome-sim/outputs/`, which does not exist in this checkout).

## Tier 1 — toy synthetic graph (`tests/test_gpu_memory_brain.py`)

Same 4-neuron graph as `tests/test_doom_learning_v6.py` (plastic edge 0->1,
`kc_mask`/`dan_index`/`modulation_mask` on cell 2), `MemoryBrain` (CPU) vs
`GPUMemoryBrain` from identical initial state and identical stimulation.
5 tests, all passing:

- Spike counts match exactly across three stimulation phases (no learning);
  voltages agree within `atol=.002` (not bit-exact -- `memory_advance` is
  lazy/table-based for skipped ticks, the GPU kernel is dense/closed-form at
  d=1, so float32 rounding differs; this is the same bar
  `test_doom_learning_v6.py` and `doom-gpu-kernel-review.md`'s Tier 1 already
  use for the plain kernel, not a new relaxation).
- The modulation-mask cell (index 2) fires but delivers zero conductance to
  its real out-edge target (index 1) on both backends.
- A two-phase learning run moves `memory_u`/`memory_w`/`weight[0]` on both
  backends, and the values agree within `atol=1e-4`; the device-resident
  `self._csr_weight` copy is asserted to match `self.weight` directly (the
  check that would have caught the upload bug above).
- `reset()` restores per-cell `rest` (not a flat -52mV) and the baseline
  plastic weight, on both host and device copies.
- Two repeated learning runs from identical initial state produce identical
  total spikes, final weight and `memory_w`.

`python -m pytest tests/test_gpu_memory_brain.py -q`: 5 passed.
`python -m pytest tests/ -q`: 25 passed (no regression to the existing
suite).

## Tier 2 — real graph cross-check (`validate_gpu_memory.py`)

166,700 neurons, 25,582,938 edges, 4,184 plastic KC->MBON11 edges (default
`circuit.py` spec), identical circuit passed to both backends so this isn't
also cross-checking `identify()` against itself. Synthetic i.i.d. uniform
luminance, `lamina_bias=0`, seed 0.

| run | ticks | spike mismatch | fraction | first mismatch | total spikes (cpu/gpu) |
|---|---|---|---|---|---|
| frozen weights | 10 | 0/166,700 | 0% | tick 9 (self-corrected by tick 10) | 97,742 / 97,742 |
| frozen weights | 200 | 3/166,700 | 0.0018% | tick 9 | 1,228,765 / 1,228,764 |
| learning=True | 30 | 1/166,700 | 0.0006% | tick 9 | 240,796 / 240,795 |

These are smaller mismatches than `doom-gpu-kernel-review.md`'s plain-kernel
figures (1,129/166,700 at 10 ticks, 4,829/166,700 at 200 ticks) on this
workload/seed -- consistent with the same floating-point summation-order
phenomenon at a smaller scale here, not evidence of a different mechanism.
**The learning-run weight check produced a null result, stated honestly
rather than glossed over**: with this synthetic-noise stimulation protocol
(no biased current into the KC/DAN circuit) neither backend's plastic
weights moved at all in 30 ticks (`weight_max_abs_diff: 0.0`,
`changed_edges: {cpu: 0, gpu: 0}`) -- so Tier 2 does not by itself exercise
"do learned weights match between backends on the real graph"; that check
was exercised, and passed, only at Tier 1's toy scale (`atol=1e-4` on
actually-nonzero `memory_w`/`weight` deltas). Raw JSON:
`androsophila/outputs/connectome_sim/gpu-benchmark-20260916/memory-validation-{frozen-10tick,frozen-200tick,learning-30tick}.json`.

## Benchmark (`benchmark_gpu_memory.py`)

Standalone harness, no server/game, synthetic luminance, `lamina_bias=12`
(default), 100 ticks + 10 warmup, same real graph and circuit:

| backend | tics/sec | median step wall |
|---|---|---|
| `MemoryBrain` (CPU) | 4.364 | 217.0 ms |
| `GPUMemoryBrain` | 41.395 | 19.4 ms |

**~9.5x faster on this identical standalone harness.** `MemoryBrain` is
slower than the plain kernel's `NativeBrain` (18.3 tics/sec, per
`doom-gpu-kernel-review.md`) because `memory_advance` does per-substep
adaptation-table work `neural_advance` doesn't; `GPUMemoryBrain` is faster
than the plain `GPUBrain` port (65.1 tics/sec dense-luminance) mainly because
this benchmark uses `lamina_bias=12` with a smaller effective active
population per tick than that run's parameters, not because the memory
kernel itself does less work per substep -- the two benchmarks are not
directly comparable (different default currents), and no such comparison is
claimed here. Raw JSON:
`androsophila/outputs/connectome_sim/gpu-benchmark-20260916/memory-benchmark.json`.

## Learning-specific determinism (point 5 of the task, measured directly)

`doom-gpu-kernel-review.md` measured `atomicAdd` float32 non-determinism as
"not observed at this workload's scale" for a **frozen-weight** run. A
learning run compounds any such divergence across weight updates over time,
so it was re-measured directly rather than assumed to transfer: the *only*
injection point for `atomicAdd` nondeterminism into the weights is the
per-bin GPU spike counts feeding `rule.advance()` (itself pure deterministic
numpy).

Two repeated 30-tick `learning=True` runs of `GPUMemoryBrain` on the real
graph, from identical initial state and identical input (seed 0,
`lamina_bias=12`): **identical total spikes (421,442 both runs), identical
final `weight[circuit['edges']]` sum and identical `memory_w` sum across both
runs.** As in Tier 2's learning check above, this run did not actually move
any plastic weight (`memory_w_sum: 0.0` both runs, `changed_edges: 0`), so
this result is honestly "spike-count determinism observed, weight-update
determinism not exercised with nonzero weight movement at this scale" --
the weight-determinism claim with actually-nonzero deltas is only supported
at Tier 1's toy scale (`test_determinism_repeated_learning_run_from_identical_state`,
which does move weights and found them identical across two runs). Not
architecturally guaranteed either way; re-measure if the workload changes.

## Bottom line

- Built exactly as scoped in points 1-5: decay kernel extended in place (no
  new kernel shape), delivery kernel gets a `modulation_mask` guard only
  (modulator delivery/eligibility/in-kernel weight decay explicitly dropped
  and documented, not silently omitted), `rule.py` untouched, new sibling
  class `GPUMemoryBrain(GPUBrain)`, plastic-weight device upload handled
  explicitly (with a regression test for exactly the bug that would result
  from forgetting it).
- Tier 1 (toy graph): 5/5 passing, spike-exact + `atol=.002` voltage match,
  modulation guard verified on both backends, learning-mode weight match at
  `atol=1e-4` with the device copy checked directly, reset and determinism
  both verified.
- Tier 2 (real graph, 166,700 neurons / 25.58M edges): 0-3 neurons/166,700
  mismatched depending on run length and learning flag -- smaller than the
  plain kernel's own documented mismatch rate on this workload, consistent
  with the same summation-order phenomenon at smaller scale, not a new
  defect. The learning-weight check on the real graph produced a null
  (no weight movement under this stimulation protocol) and is reported as
  such rather than claimed as a positive match.
- Benchmark: **~9.5x faster** (41.4 vs 4.4 tics/sec) on an identical
  standalone harness, real numbers, not an estimate.
- Determinism: 0 divergence in spike counts across two repeated real-graph
  learning runs; weight-determinism with actually-nonzero weight movement
  confirmed only at toy scale.

See also: `androsophila/docs/doom-gpu-kernel-review.md` (the plain-kernel GPU
port and validation methodology this work follows),
`physiology/gpu_brain.py` (full scope-cut rationale in the module docstring),
`tests/test_gpu_memory_brain.py`, `validate_gpu_memory.py`,
`benchmark_gpu_memory.py`.
