# connectome-sim constraints

- Read `../connectome-lab/REPRODUCIBILITY.md` before changing anything that
  computes `ROOT` from `__file__`, or anything touching how consumers check
  this repo out: this is the shared engine, checked out as a real git
  submodule pinned to a tagged commit in every harness's `connectome_sim`
  path (`androsophila`, `flappy-haltere`, `flybody-connectome`). Edit it only
  in this canonical checkout, never through a harness's `connectome_sim/`
  path — that path is a separate checkout, and an edit made there is
  invisible to git everywhere else until committed and pushed from the
  harness (don't). Tag and push a release here first, then bump every
  consumer's pin in the same session; details and the `.absolute()`-not-
  `.resolve()` requirement are in `REPRODUCIBILITY.md`, not duplicated
  per-repo.
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
