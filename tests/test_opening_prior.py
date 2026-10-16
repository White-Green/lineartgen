import jax
import jax.numpy as jnp
import numpy as np
from opening_prior_objective import disk_reduce, opening_consistency, soft_opening


def example():
    yy, xx = jnp.meshgrid((jnp.arange(128) + 0.5) / 4, (jnp.arange(128) + 0.5) / 4, indexing="ij")
    main = abs(yy - 12) - 1
    narrow = abs(yy - 20) - 0.2
    return main[None, ..., None], narrow[None, ..., None], jnp.ones((1, 32, 32, 1))


def test_narrow_extra_ink_penalty_and_gradient():
    main, narrow, t = example()
    pred = jnp.minimum(main, narrow)
    target = main[:, None]

    def fn(p):
        return opening_consistency(p, target, t)

    value, grad = jax.value_and_grad(fn)(pred)
    assert float(value) > 0
    assert np.isfinite(grad).all()
    assert float(jnp.linalg.norm(grad)) > 0
    assert float(fn(pred - 0.05 * grad)) < float(value)
    assert float(jnp.clip(0.5 - 4 * (pred - 0.05 * grad), 0, 1).sum()) < float(
        jnp.clip(0.5 - 4 * pred, 0, 1).sum()
    )
    assert float(fn(main)) == 0


def test_teacher_taper_and_candidate_exemption():
    main, narrow, t = example()
    x = (jnp.arange(128) + 0.5) / 4
    taper = jnp.minimum(main, narrow - (x[None, None, :, None] - 16) * 0.04)
    teacher = jnp.stack((main, taper), 1)
    assert float(opening_consistency(taper, teacher, t)) == 0
    gradient = jax.grad(lambda p: opening_consistency(p, teacher, t))(taper)
    np.testing.assert_array_equal(gradient, jnp.zeros_like(gradient))
    assert float(opening_consistency(main, teacher, t)) == 0


def test_inactive_neighborhood_and_blank():
    main, narrow, t = example()
    pred = jnp.minimum(main, narrow)
    t = t.at[:, 18:].set(0)
    assert float(opening_consistency(pred, main[:, None], t)) == 0
    assert float(opening_consistency(pred, main[:, None], jnp.zeros_like(t))) == 0
    blank = jnp.ones_like(main)
    assert float(opening_consistency(blank, blank[:, None], jnp.ones_like(t))) == 0
    assert float(opening_consistency(narrow, blank[:, None], jnp.ones_like(t))) > 0


def test_disk_d4_subset_idempotence():
    a = jnp.asarray(np.random.default_rng(3).uniform(size=(2, 17, 17, 1)), dtype=jnp.float32)
    opened = soft_opening(a)
    assert np.all(np.asarray(opened) <= np.asarray(a))
    np.testing.assert_array_equal(soft_opening(opened), opened)
    for reflected in (False, True):
        for k in range(4):

            def transform(v, reflected=reflected, k=k):
                return jnp.rot90(jnp.flip(v, axis=2) if reflected else v, k, axes=(1, 2))

            np.testing.assert_array_equal(soft_opening(transform(a)), transform(opened))
    np.testing.assert_array_equal(disk_reduce(jnp.ones_like(a), maximum=False), jnp.ones_like(a))
