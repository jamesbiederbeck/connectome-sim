# connectome-sim constraints

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
