import jax
import jax.numpy as jnp
import numpy as np
from field_repair_study import OPTIMIZER, objective, step


def test_wrong_sign_gradients_and_protection():
    t = jnp.ones((1, 1, 32, 32, 1))
    t = t.at[:, :, 12:20, :, :].set(-1)
    field = jnp.ones((1, 32, 32, 1))
    field = field.at[:, 12:20, :, :].set(-1)
    field = field.at[:, 14:18, 18:22, :].set(0.7)
    field = field.at[:, 3:7, 18:22, :].set(-0.7)
    strength = jnp.ones((1, 8, 8, 1)).at[:, :, :2, :].set(0)
    fields, targets, strengths = (field,), (t,), (strength,)
    (_, _), grads = jax.value_and_grad(objective, has_aux=True)(fields, targets, strengths)
    g = np.asarray(grads[0])
    assert g[0, 15:17, 19:21, 0].min() > 0
    assert g[0, 4:6, 19:21, 0].max() < 0
    np.testing.assert_array_equal(g[:, :, :8], 0)
    updated, _, _ = step(fields, OPTIMIZER.init(fields), targets, strengths)
    assert float(objective(updated, targets, strengths)[0]) < float(
        objective(fields, targets, strengths)[0]
    )
    np.testing.assert_array_equal(updated[0][:, :, :8], field[:, :, :8])


def test_all_recursive_fields_receive_gradients_and_replay():
    fields = tuple(jnp.full((1, 32 // (2**i), 32 // (2**i), 1), 0.5) for i in range(3))
    targets = tuple(jnp.full((1, 1, *f.shape[1:]), -1.0) for f in fields)
    strengths = tuple(jnp.ones((1, f.shape[1] // 4, f.shape[2] // 4, 1)) for f in fields)
    (_, _), grads = jax.value_and_grad(objective, has_aux=True)(fields, targets, strengths)
    assert all(np.linalg.norm(g) > 0 for g in grads)
    a = step(fields, OPTIMIZER.init(fields), targets, strengths)
    b = step(fields, OPTIMIZER.init(fields), targets, strengths)
    jax.tree.map(np.testing.assert_array_equal, a, b)
