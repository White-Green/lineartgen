# BR: settle the point-velocity model after6000 updates

AO's lower learning rate helped fixed one-pass fitting. The existing BE low-rate
continuation used interval velocity, not BC's exact point MSE. Test point MSE
with a matched continuation control before declaring its fitting limit.

Restore BC's actual MSE6000 model, AdamW moments/count and EMA into two separate
arms. Both receive1200 further updates, steps6000–7199, identical prepared
batches and cyclic recursion1–6. Control keeps actual LR1e-4; settled uses1e-5.
Base rates5e-5/5e-6 times sqrt(batch4). No warmup or decay during continuation.
No optimizer/EMA reset. All other BC settings remain:130628 parameters, same
fixed4 TRAIN/4 unfitted PROBE512px geometry, Gaussian coefficient.25 after resize,
25%tau0 otherwise sin²(pi*U/2), point MSE, scale weight0 and recursive gradients,
training seed2026101093,32-step sin² Euler. BQ auxiliary loss is not used.

Before updates reproduce all25 frozen BC MSE6000 prediction blocks bitwise,
including partial mask. Both initial payloads must match the parent's entire
restored state. First6 control updates must match original BC training functions
and full state bitwise; replay both new arms' first6 updates independently.
States6000/6006/6600/7200,16 raw/EMA exports, all-state/export equality checks.
Tests include cloned moments/EMA, different learning rates with equal initial
gradient statistics, diverging updates, and full restoration/next-update equality.

Primary finalraw7200 only. Same8 cases,depth1/3/6,4 fixed fresh noise draws, plus
initialdepth3/draw0:200 rows,52 physical/clipped blocks,88 fit criteria per arm,
64 retention conditions (settled vs continued control). Review all5 native/
overview/partial sheets, all four draws. Record tau0 counts for both arms and
assert they agree. No threshold relaxation or checkpoint/draw/EMA selection.

This isolates the late learning rate at equal update count. A comparison to
BC6000 additionally changes duration. No production/plugin/default change or
adoption; fitting gains would still require broader synthetic training,
independent/sealed confirmation and actual runtime parity.
