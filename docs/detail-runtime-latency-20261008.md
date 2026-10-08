# CPU runtime cost of the detail head

The width16 detail branch adds measurable runtime cost: for512×512 BGRA inputs,
median CPU latency rises from190 to252ms for one iteration, and from799 to1,075ms
for five. These measurements use the actual local1200 raw checkpoints, not the
still-training broader candidates. They measure speed only, not quality.

The saved plan fixes one ONNX Runtime CPU thread, three warmups and15 recorded
repeats per model/shape/iteration count, alternating model order each repeat.
Timing covers the complete `LineartModel.infer` call, including image preparation,
noise, recursive forwarding and BGRA output. Session creation, warmups and output
hashing are excluded. All repeated output hashes are identical within a condition.

| Input height×width | Iterations | Guided control median ms | Detail16 median ms | Ratio |
| --- | ---: | ---: | ---: | ---: |
|145×129|1|17.02|21.96|1.290|
|145×129|2|18.55|24.29|1.310|
|145×129|5|68.56|100.45|1.465|
|512×512|1|189.71|251.83|1.327|
|512×512|2|355.08|443.66|1.249|
|512×512|5|798.86|1,074.61|1.345|
|513×385|1|245.46|304.63|1.241|
|513×385|2|472.60|541.53|1.146|
|513×385|5|1,043.05|1,277.36|1.225|
|1,024×1,024|1|723.24|906.80|1.254|
|1,024×1,024|2|1,523.74|1,785.69|1.172|
|1,024×1,024|5|3,660.13|4,353.20|1.189|

Hardware is AMD Ryzen AI Max+395, Linux, Python3.12.3 and ONNX Runtime1.29.0.
GPU training was running concurrently, with host preprocessing, so these are
provisional application timings rather than an isolated hardware benchmark.
The small-input timings in particular show substantial variability; retain the
individual observations and p95 values instead of inferring exact linear scaling.
Runtime padding and pyramid sizes are recorded for every condition. These numbers
do not establish Windows DirectML or Krita GUI performance.

There are432 calls including warmups, of which360 are timed observations.
The control export passes the exporter’s three dynamic-shape JAX comparisons;
the detail export previously passed27 actual BGRA/JAX comparisons. Model, config,
script, runtime and plan hashes are retained under
`runs/detail-runtime-latency-20261008/`. No production model is changed.
