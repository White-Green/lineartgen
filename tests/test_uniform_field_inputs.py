import jax.numpy as jnp
import numpy as np
from opaque_geometry import geometry, render
from uniform_field_inputs import uniform_plan


def test_uniform_teacher_makes_thin_crossing_opaque_without_moving_geometry():
    metadata = dict(
        path_geometry=[
            dict(points=[[2.5, 8.5], [13.5, 8.5]], widths=[4.0, 4.0]),
            dict(points=[[8.5, 2.5], [8.5, 13.5]], widths=[0.4, 0.4]),
        ]
    )
    p = dict(scene=dict(metadata=metadata), probe=dict(metadata=metadata))
    q = uniform_plan(p)
    old_args = geometry(metadata, size=16, margin=0)
    new_args = geometry(q["scene"]["metadata"], size=16, margin=0)
    np.testing.assert_array_equal(old_args[0], new_args[0])
    assert p["scene"]["metadata"]["path_geometry"][1]["widths"] == [0.4, 0.4]
    old = np.asarray(render(*(jnp.asarray(x) for x in old_args), size=16))
    new = np.asarray(render(*(jnp.asarray(x) for x in new_args), size=16))
    assert old[3:8, 8, 0].min() > -0.5
    assert new[3:13, 8, 0].max() == -1
    assert new[0, 0, 0] == 1
