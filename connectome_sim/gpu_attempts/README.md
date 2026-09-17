# GPU kernel attempts (historical)

Three GPU delivery-kernel designs were built and measured while developing
`doom/gpu.py`. Only the last is in `doom/gpu.py`/wired into
`tests/test_doom_reference.py`; the first two are preserved here for
reference, not used by anything.

| | file | design | measured | % of ~192 GB/s peak |
|---|---|---|---|---|
| 1 | `v1_spmv.py` (`GPUBrainSpMV`) | cuSPARSE SpMV | 1.509 tics/sec | 49% |
| 2 | `v2_thread_per_row.py` (`GPUBrainThreadPerRow`) | one CUDA thread per destination-neuron row | 0.892 tics/sec | 27% |
| 3 | `../gpu.py` (`GPUBrain`, active) | one CUDA warp per destination-neuron row | 2.064 tics/sec | 56% |

All three passed the exact Brian2-oracle validation
(`tests/test_doom_reference.py`) at small scale; none beats `NativeBrain`'s
~13-15 tics/sec on this hardware, because the underlying dense-every-substep
formulation is bandwidth-bound at this graph size regardless of kernel
quality (see `docs/doom-gpu-kernel-review.md` for the roofline analysis and
the event-driven redesign that would actually be needed to beat the CPU).

These two files are not imported by `doom/gpu.py` or by any test; they exist
purely so the discarded designs aren't lost. They do reuse
`doom.gpu._configure_cuda_env` (the CUDA library preload helper), since that
part didn't change across attempts.
