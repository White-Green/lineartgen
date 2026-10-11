import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

rows = []
for shape in [(2, 32, 32, 1), (2, 32, 32, 16), (4, 256, 256, 1), (4, 16, 16, 40)]:
    b, h, w, c = shape
    out = (b, 2 * h, 2 * w, c)

    def f(x, weights, out=out):
        return (jax.image.resize(x, out, "nearest") * weights).sum()

    grad = jax.jit(jax.grad(f))
    x = jnp.zeros(shape)
    for pattern in ["ones", "integer"]:
        weights = (
            np.ones(out, np.float32)
            if pattern == "ones"
            else (np.arange(np.prod(out)).reshape(out) % 7).astype(np.float32)
        )
        expected = weights.reshape(b, h, 2, w, 2, c).sum((2, 4))
        got = np.asarray(grad(x, jnp.asarray(weights)))
        row = dict(
            shape=shape,
            pattern=pattern,
            exact=bool(np.array_equal(got, expected)),
            max_abs=float(np.abs(got - expected).max()),
            wrong=int(np.count_nonzero(got != expected)),
            first_actual=got.ravel()[:8].tolist(),
            first_expected=expected.ravel()[:8].tolist(),
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
result = dict(platform=jax.devices()[0].platform, flags=os.getenv("XLA_FLAGS", ""), rows=rows)
Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")
