import jax
import jax.numpy as jnp
import numpy as np
from evaluate_connection_queries import widest
from procedural_queries import Queries, generate, patches, query_loss

from lineartgen.training.connectivity import connection_capacity


def test_fresh_geometry_reproducible_and_teacher_queries_match_oracle():
    run = jax.jit(jax.vmap(connection_capacity))
    all_kinds = set()
    capacities = []
    first = generate(2026100951, 0)
    repeat = generate(2026100951, 0)
    jax.tree.map(np.testing.assert_array_equal, first[:2], repeat[:2])
    assert first[2] == repeat[2]
    assert not np.array_equal(first[0], generate(2026100951, 1)[0])
    for step in range(3):
        clean, queries, records = generate(2026100951, step)
        assert clean.shape == (4, 512, 512, 1)
        assert np.isfinite(clean).all() and clean.min() >= -1 and clean.max() <= 1
        all_kinds.update(r["kind"] for r in records)
        p = np.asarray(patches(jnp.asarray((1 - clean) / 2), queries))
        a, b = (x.reshape(-1, 64, 64) for x in (queries.source, queries.sink))
        values = run(p, a, b)
        for image, source, sink, value in zip(p, a, b, values, strict=True):
            assert float(value) == widest(image, source, sink)
            if source.any():
                capacities.append(float(value))
        loss, _ = query_loss(
            jnp.asarray(clean), jnp.asarray(clean), jnp.ones_like(clean, dtype=bool), queries
        )
        assert loss == 0
    assert {"thin", "corner", "gap", "parallel", "crossing", "junction"} <= all_kinds
    assert any(x == 0 for x in capacities) and any(x > 0 for x in capacities)


def test_class_normalization_and_protected_gradient():
    target = np.ones((2, 64, 64, 1), np.float32)
    target[:, 32, :, 0] = -0.2  # opacity.6 antialiased stroke
    target[1, 32, 32, 0] = 1
    predicted = target.copy()
    predicted[0, 32, :, 0] = 0.2  # opacity.4
    predicted[1, 32, 32, 0] = 0.5  # opacity.25 bridge
    a, b = (np.zeros((2, 1, 64, 64), bool) for _ in range(2))
    a[:, 0, 32, 8] = True
    b[:, 0, 32, 56] = True
    queries = Queries(np.zeros((2, 1, 2), np.int32), a, b)
    active = np.ones_like(target, bool)
    def loss(p, edit):
        return query_loss(p, jnp.asarray(target), edit, queries)[0]
    value, grad = jax.jit(jax.value_and_grad(loss))(jnp.asarray(predicted), active)
    np.testing.assert_allclose(value, 0.5 * (0.2**2 + 0.25**2), atol=1e-7)
    assert np.isfinite(grad).all() and np.count_nonzero(grad) == 2
    assert grad[1, 32, 32, 0] < 0  # descent makes a false bridge whiter
    active[1, 32, 32, 0] = False
    _, grad = jax.jit(jax.value_and_grad(loss))(jnp.asarray(predicted), active)
    assert grad[1, 32, 32, 0] == 0
