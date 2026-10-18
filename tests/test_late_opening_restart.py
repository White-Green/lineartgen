import json

import jax.numpy as jnp
import numpy as np
from cue_capacity import make_arm
from learned_coarse_axis_model import load_field
from late_opening_restart_study import HH, HJ, HM, STEPS, restart_config


def test_restart_schedule_and_sqrt_batch_scaling():
    model, cfg = load_field(HJ / "snapshots/step-18000/consistent/raw")
    cfg = restart_config(cfg)
    arm = make_arm(cfg, model)
    assert cfg.train.steps == cfg.train.learning_rate_decay_steps == 6000
    assert cfg.train.peak_learning_rate == 1e-4 * np.sqrt(cfg.train.batch_size) == 2e-4
    np.testing.assert_allclose(
        arm["schedule"](jnp.array([0, 3000, 6000, 7000])),
        [2e-4, 1.1e-4, 2e-5, 2e-5],
        atol=1e-10,
        rtol=1e-6,
    )
    assert int(arm["optimizer"].step) == 0
    larger_batch = cfg.train.model_copy(update=dict(batch_size=16))
    assert larger_batch.peak_learning_rate == 4e-4


def test_duration_control_uses_identical_full_batch_order():
    hh = [json.loads(s) for s in (HH / "training/sampling.jsonl").read_text().splitlines()]
    hm = [json.loads(s) for s in (HM / "training/sampling.jsonl").read_text().splitlines()]
    assert len(hh) == len(hm) == STEPS
    for a, b in zip(hh, hm, strict=True):
        for key in (
            "input_sha256",
            "candidate_payload_sha256",
            "context_keep",
            "audit",
            "coherent_extension",
            "consistent_original",
        ):
            assert a[key] == b[key]
