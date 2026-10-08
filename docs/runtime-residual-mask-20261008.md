# Exact protected pixels in the residual ONNX sampler

The NumPy sampler could change zero-noise values by 5.96e-8 through roundoff in
the multistep update. JAX already explicitly retains these pixels for residual
models. The runtime now applies the same guard at every pyramid level and
iteration, enabled only when ONNX metadata declares residual output. Direct-output
models retain their previous behavior. The external BGRA API is unchanged.

Seven regression cases cover partial and complete identity masks at 2/5/20
iterations, all three pyramid levels, unchanged active-pixel updates for a
pointwise oracle, and legacy direct-output behavior. Together with existing
sampling, packaged runtime, and source-conditioning tests: **31 passed in 42.04s**
(`/tmp/runtime-residual-mask-tests-20261008.log`).

The actual selected alpha=1/16 EMA checkpoint was exported and compared again
with CPU JAX through the BGRA API. All 27 cases passed; protected floating-point
error is now exactly zero. Maximum continuous discrepancy remains 1.7643e-5,
maximum byte discrepancy is one alpha level, and five bytes differ across all
outputs. This does not identify a cause of perceptible gray halos; it corrects a
small consistency issue. Windows DirectML and Krita GUI were not run here.

Artifacts: `runs/corruption-blend-runtime-exact-mask-20261008/` contains the ONNX,
export checks, and complete runtime comparison records. The training checkout
and its pinned runtime files were not modified. Production model assets remain
unchanged pending second-seed and prospective synthetic confirmation.
