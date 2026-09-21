# connectome-sim constraints

- Read `../connectome-lab/REPRODUCIBILITY.md` before changing anything that
  computes `ROOT` from `__file__`, or anything touching how consumers check
  this repo out (submodule vs. symlink): this is the shared engine, checked
  out via a symlink from every harness's `connectome_sim` path, and both the
  symlink setup and the `.absolute()`-not-`.resolve()` requirement it implies
  are documented there, not duplicated per-repo.
- This repository is the reusable connectome engine only: connectome import,
  the native/GPU LIF kernels, photoreceptor sampling, and the generic
  dopamine-gated-plasticity physiology code. It has no game harness of its
  own and no launch/spectator requirements — those live in the repos that
  consume this one as a submodule.
- Retain every released connection between the retained neuronal entries. Do
  not crop circuits, prune weak/self edges, or replace the network with a
  game policy. This applies regardless of which consumer imports it.
- Document measured circuitry, inferred mappings, chosen dynamics and
  unresolved mechanisms separately. A full retained connectome is not a
  literal living brain.
- Preserve failed experiments and controls. Changed weights and long
  survival alone do not establish learning; numerical tests are not
  biological validation.
- Store credentials and machine-specific origins only in ignored
  configuration.
- Preserve third-party notices. Do not bundle external research workbooks,
  papers, or unrelated projects in archives.
- The native kernel (`kernel.cpp`) and GPU backend (`gpu.py`) must stay
  numerically equivalent to each other; any GPU-backend change belongs here,
  not in a consumer repo, since this is also the basis of the open PR back to
  `nftechie/doomfly`.

- Graph and kernel artifacts live under the consuming repo's
  `outputs/connectome_sim/`, not `outputs/doom/`. The old name was a DOOMFLY
  leftover; this engine is not Doom-specific and is consumed by several
  harnesses. The kernel path override is `CONNECTOME_KERNEL_PATH`.
- `prepare.py` belongs here, not in a harness: it is the second half of the
  import pipeline that `engine.py` validates the output of. Keep its retinal
  projection and readout selection parameterised so no consumer has to fork it.
