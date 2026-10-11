# CE: separate resolution order from endpoint capability, without training

Registered while CD's step-count evaluation runs. Keep CC final raw7200 control
and band134688-parameter models, same512px inputs, masks, Gaussian keys and32
uniform steps. No use of CD's eventual preferred step count and no new weights.

Compare three explicitly different conditions:

* `recursive`: existing time-first sampler. At each time, compute endpoints from
  small to large; integrate every resolution's independently evolving state.
* `cascade`: complete all32 steps at the smallest resolution first. Clip its
  completed output as the fixed insert for all32 steps of the next resolution,
  and continue to the largest. Each stage starts from the same independently
  noised source as before. Information still flows only small to large; a larger
  output is never fed back into a smaller model. Same32*depth network calls.
* `oracle`: diagnostic teacher intervention, never a deployable mode. Run only
  the largest resolution's32 steps with its immediate smaller clean teacher
  pyramid image as a fixed insert (zero insert for depth1). Source/noisy state,
  strength and progress are unchanged. This exposes teacher information and
  cannot count as a learned quality gain. It asks whether providing the correct
  smaller endpoint helps this existing model at all; conditioning is out of the
  model's generated-insert distribution, so improvement is not a complete cause
  attribution or a guarantee of a trainable implementation.

At each stage use rawD=x+sqrt(t)*head, boundedD=clip(rawD), v=(boundedD-x)/u,
x+=v/32, with protected pixels reset to original source. The32 nodes all have
u>=1/32, matching the existing time floor exactly. Smaller inserts are clipped
before passing upward. Noise is added independently after each source resize;
no enlarged output is substituted for the informative input image. The oracle
uses the pyramid of where(mask,teacher,source). Non-oracle APIs reject a teacher.

Freeze code/plan/data/model hashes before evaluation. Replay both original
recursive references:50 physical/clipped blocks and192 metric rows bitwise.
CPU tests check per-resolution call counts/order, constant completed inserts,
no teacher accepted by cascade/recursive, same-depth1 outputs within1e-6, oracle
teacher influence, protected pixels and finite results. The1e-6 depth1 allowance
accounts for possible fusion differences in the new single-stage compiled head;
it is not a relaxed tolerance for replay of the unchanged recursive baseline.

Evaluate2 models x3 modes x3 depths1/3/6 x4 draws x8 cases:576 rows and150 blocks
including partial depth6 batches. Keep88 fitting checks/method(528 total), and
64 F1/deficit retention checks each for cascade/oracle versus recursive per model
(256 total). Label oracle results explicitly in every table/figure. Ordinary
fitting criteria applied to an oracle do not establish deployability.

Inspect10 fixed sheets: train/probe native and overview per model, all4 draws;
partial native with one draw per four train cases. Unchanged AD crops, all
outputs saved. No new shapes, training seed, evaluation seed, sealed confirmation,
production model, plugin, runtime default, export or deployment.
