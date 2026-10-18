import json

import jax.numpy as jnp
import numpy as np
from cue_capacity import make_arm
from late_opening_duration_study import HH, HL, METHODS, START, STEPS
from learned_coarse_axis_model import load_field


def test_inherited_schedule_stays_flat_past_original_endpoint():
    for name in METHODS:
        model, cfg = load_field(HL / f"snapshots/step-{START}/{name}/raw")
        assert cfg.train.steps == cfg.train.learning_rate_decay_steps == START
        assert cfg.train.peak_learning_rate == 1e-5 * np.sqrt(cfg.train.batch_size) == 2e-5
        arm = make_arm(cfg, model)
        np.testing.assert_allclose(
            arm["schedule"](jnp.arange(START, STEPS + 1)), 2e-5, atol=1e-10, rtol=0
        )


def test_archived_prefix_and_remaining_source_order():
    old = [json.loads(s) for s in (HL / "training/sampling.jsonl").read_text().splitlines()]
    all_rows = [json.loads(s) for s in (HH / "training/sampling.jsonl").read_text().splitlines()]
    assert len(old) == START and len(all_rows) == STEPS
    for a, b in zip(old, all_rows[:START], strict=True):
        for key in (
            "input_sha256",
            "candidate_payload_sha256",
            "context_keep",
            "audit",
            "coherent_extension",
            "consistent_original",
        ):
            assert a[key] == b[key]
    remaining = all_rows[START:]
    counts = {
        name: sum(r["audit"]["family"] == name for r in remaining)
        for name in ("legacy", "long_bar_fill", "ambiguous")
    }
    assert counts == dict(legacy=3000, long_bar_fill=1200, ambiguous=600)
    assert [r["step"] for r in remaining] == list(range(START + 1, STEPS + 1))
