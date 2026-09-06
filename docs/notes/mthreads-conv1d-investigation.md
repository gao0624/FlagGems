# MTHREADS Conv1D Optimization Investigation Notes

This branch documents the empirical investigation of optimizing the MTHREADS
MUSA Conv1D operator on MTT S5000 (60 SMs, CC 3.1).

## Method

* Test machine: 8x MTT S5000, MUSA 4.3.5, torch_musa 2.7.1, PyTorch 2.7.1,
  Triton 3.6.0
* Benchmark: `tests/test_conv1d.py` and `benchmark/test_conv1d.py`,
  `--mode kernel` (uses `triton.testing.do_bench` with MUSA event-timing
  fallback). Warmup=200, iter=500.
* Baseline: `flag_gems.runtime.backend._mthreads.ops.conv1d` is not registered
  on `upstream/master`; the MTHREADS vendor therefore falls through to
  `flag_gems.ops.conv1d` (the unsqueeze-to-2D `conv2d` wrapper). Numbers
  below are reported relative to that baseline.

## Result

* The current upstream MTHREADS Conv2D-via-unsqueeze path is the
  hardware-best path on MTT S5000 for the full set of official
  `tests/test_conv1d.py` shapes (28/28 PASS, FP16/FP32,
  conv1d / conv1d_padding / conv1d_dilation, strides, dilations,
  groups, str-padding 'valid' / 'same', int paddings).
* A dedicated 1D implicit-GEMM Triton kernel with a per-shape
  `USE_TF32X3` dispatch was tried. Empirical ratios on the official
  `benchmark/test_conv1d.py -m conv1d_padding` shapes (13 FP16 +
  13 FP32, kernel_mode, warmup=200, iter=500):
  - On the K3/K5 / FP32 / weight_c >= 48 cases the dedicated kernel
    is 0.78-0.86x of the baseline (slower).
  - On the K7g2/K11 / weight_c <= 16 / FP16 cases the dedicated kernel
    with `tf32x3` is 0.45-2.5x of the baseline (mixed; the wins on
    K11 same are offset by regressions on K7g2 p3 / p6 / same).
  - K3/K5 FP16 dedicated kernels are 0.13-0.50x of the baseline.
  Net: the dedicated kernel does not produce a clean Pareto win
  against the existing MTHREADS Conv2D vendor kernel on the current
  official test set, and any kept win class comes with corresponding
  regression risk on the remaining shapes.
* No production-quality change is landed in this branch. The
  investigation is recorded here for future reference.

## Vendor override

This branch contains no functional changes to the MTHREADS vendor
path. The local `flag_gems.runtime.backend._mthreads.ops.conv1d`
file from earlier iterations was removed before this commit so that
the public dispatch is identical to `upstream/master`.
