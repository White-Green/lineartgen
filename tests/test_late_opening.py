import jax
import jax.numpy as jnp
import numpy as np
from coherent_broad_base_objective import axis_parts as old_parts
from coherent_broad_objective import axis_parts
from cue_capacity import make_arm
from late_opening_study import HJ, STEPS
from learned_coarse_axis_model import load_field
from test_consistent_broad import toy


def test_zero_weight_preserves_original_costs_and_gradients():
    f, fam, t, n = toy()
    eligible = jnp.array([False, True])
    for allowed in (jnp.zeros(2, bool), jnp.array([True, False])):
        args = (*fam, t, eligible, allowed, n)
        old = old_parts(f, *args, policy="retained")
        new = axis_parts(f, *args, policy="retained", opening_weight=0.0)
        for k in old:
            np.testing.assert_array_equal(old[k], new[k])
        a = jax.grad(lambda ff, args=args: old_parts(ff, *args, policy="retained")["loss"])(f)
        b = jax.grad(
            lambda ff, args=args: axis_parts(ff, *args, policy="retained", opening_weight=0.0)[
                "loss"
            ]
        )(f)
        jax.tree.map(np.testing.assert_array_equal, a, b)


def test_opening_addition_preserves_assignment_and_uses_the_existing_prior():
    from test_opening_prior import example

    main, narrow, strength = example()
    f = (jnp.minimum(main, narrow),)
    teacher = (jnp.repeat(main[:, None], 9, axis=1),)
    args = (
        teacher,
        teacher,
        teacher,
        (strength,),
        jnp.array([False]),
        jnp.array([True]),
        jnp.array([[0.0, 1.0]]),
    )
    a = axis_parts(f, *args, policy="retained", opening_weight=0.0)
    b = axis_parts(f, *args, policy="retained", opening_weight=0.1)
    for k in a:
        if k != "loss":
            np.testing.assert_array_equal(a[k], b[k])
    assert float(a["opening_consistency"]) > 0
    np.testing.assert_allclose(b["loss"] - a["loss"], 0.1 * a["opening_consistency"], atol=1e-7)
    grad = jax.grad(
        lambda ff: axis_parts(ff, *args, policy="retained", opening_weight=0.1)["loss"]
    )(f)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(grad))


def test_constant_continuation_rate_preserves_sqrt_batch_rule():
    model, cfg = load_field(HJ / "snapshots/step-18000/consistent/raw")
    cfg = cfg.model_copy(
        update=dict(
            train=cfg.train.model_copy(
                update=dict(
                    steps=STEPS,
                    learning_rate_decay_steps=STEPS,
                    learning_rate=1e-5,
                    minimum_learning_rate=2e-5,
                    warmup_steps=0,
                )
            )
        )
    )
    arm = make_arm(cfg, model)
    assert cfg.train.peak_learning_rate == 1e-5 * np.sqrt(cfg.train.batch_size) == 2e-5
    np.testing.assert_allclose(arm["schedule"](jnp.arange(STEPS + 1)), 2e-5, atol=1e-10, rtol=0)
    assert int(arm["optimizer"].step) == 0
