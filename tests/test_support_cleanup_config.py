import json

import jax.numpy as jnp
import numpy as np
from cue_capacity import make_arm
from learned_coarse_axis_model import load_field
from pooled_axis_study import HH, HO, STEPS, restart_config


def test_cleanup_schedule_and_actual_batch_reference_prefix():
    model, cfg = load_field(HO / "snapshots/step-6000/consistent/raw")
    cfg = restart_config(cfg)
    arm = make_arm(cfg, model)
    assert cfg.train.steps == cfg.train.learning_rate_decay_steps == STEPS == 1200
    assert cfg.train.peak_learning_rate == 1e-4 * np.sqrt(cfg.train.batch_size) == 2e-4
    np.testing.assert_allclose(
        arm["schedule"](jnp.array([0, 600, 1200])), [2e-4, 1.1e-4, 2e-5], atol=1e-10, rtol=1e-6
    )
    assert int(arm["optimizer"].step) == 0
    aa = [json.loads(s) for s in (HH / "training/sampling.jsonl").read_text().splitlines()]
    bb = [json.loads(s) for s in (HO / "training/sampling.jsonl").read_text().splitlines()]
    for a, b in zip(aa[:STEPS], bb[:STEPS], strict=True):
        for k in (
            "input_sha256",
            "candidate_payload_sha256",
            "context_keep",
            "audit",
            "coherent_extension",
            "consistent_original",
        ):
            assert a[k] == b[k]
