# Noise-convergence continuation preparation

The [4,000-update continuation](noise-convergence-plan-20261009.md) is running from numerical pin `70a98b96ba6c31d2bf505066c4c70a6afbce5006` (246 files,7 settings). No model, loss, data or inference-core code changed. The104 core/test files still match the prior298-pass manifest; the three changed/new orchestration scripts pass Ruff and compilation.

Both isolated checkpoint copies restore every model, optimizer and EMA state leaf exactly to the completed fresh-noise800 parent. All24 cached case/depth predictions from the restored models match the parent bitwise. Configured schedule horizon remains800, with actual LR at800/1,600/4,000 verified as1e-5; optimizer counters and moments were not reset. Noise generation retains the global step and case keys.

The800→806 preflight is finite and restores the entire state exactly. Actual801 and806 both restore exactly; actual806 matches the entire preflight806 state bitwise. The next endpoints are1,600 and4,000, with the latter fixed as primary. All existing independent-noise draws are now observed diagnostic/development conditions, not fresh confirmation or new clean images.

Results will include all32 primary noise-draw conditions, the old-transfer draw, all24 cached recursion conditions, five native sheets, and the existing width/connectivity/gap-opacity audit. There is no adoption from this single-case continuation. The deployed model and sealed fresh confirmation remain unchanged.
