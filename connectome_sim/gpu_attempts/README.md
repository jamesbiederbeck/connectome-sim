# GPU kernel attempts (historical)

Four GPU delivery-kernel designs were built and measured while developing
`doom/gpu.py`. Only the last (event-driven) is in `doom/gpu.py`/wired into
`tests/test_doom_reference.py`; the first three are preserved here for
reference, not used by anything.

| | file | design | measured | % of ~192 GB/s peak |
|---|---|---|---|---|
| 1 | `v1_spmv.py` (`GPUBrainSpMV`) | cuSPARSE SpMV, dense every-substep | 1.509 tics/sec | 49% |
| 2 | `v2_thread_per_row.py` (`GPUBrainThreadPerRow`) | one CUDA thread per destination-neuron row, dense every-substep | 0.892 tics/sec | 27% |
| 3 | `v3_dense_warp_per_row.py` (`GPUBrainDenseWarpPerRow`) | one CUDA warp per destination-neuron row, dense every-substep | 2.064 tics/sec | 56% |
| 4 | `../gpu.py` (`GPUBrain`, active) | event-driven: dense elementwise decay/threshold/reset + scatter delivery only for spiking neurons | see `docs/doom-gpu-kernel-review.md` | n/a (not bandwidth-bound by construction) |

All four passed the exact Brian2-oracle validation
(`tests/test_doom_reference.py`) at small scale. The first three are all
**dense-every-substep**: they read all 25,582,938 edges every 0.1ms substep
regardless of how many neurons actually spiked (~62/substep), which caps
them at ~2-3.3 tics/sec on this hardware regardless of kernel quality (see
`docs/doom-gpu-kernel-review.md` for the roofline analysis). Design #4
matches the CPU kernels' (`kernel.cpp`, `doom/engine.py`) event-driven
behavior instead — touching edges only for neurons that actually spiked —
removing that ceiling; see the doc for its measured result against
`NativeBrain`.

These three files are not imported by `doom/gpu.py` or by any test; they
exist purely so the discarded designs aren't lost. They do reuse
`doom.gpu._configure_cuda_env` (the CUDA library preload helper), since that
part didn't change across attempts.
