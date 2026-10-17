"""Small fixed varied-axis fitting set; independent synthetic transfer cases."""

import numpy as np
from ambiguous_shape_data import render, validate
from consistent_axis_objective import preferred_original


def records(split):
    assert split in ("train", "transfer")
    rng = np.random.default_rng(2026101708 if split == "train" else 2026101709)
    specs = (
        [
            (w, angle, ratio)
            for w, ratio in ((32, 0.8), (48, 1.0), (72, 1.25))
            for angle in (0, 30, 60, 90)
        ]
        if split == "train"
        else [
            (w, a, ratio)
            for w in (40, 64)
            for a, ratio in ((-15, 0.9), (15, 1.0), (45, 1.0), (75, 1.1))
        ]
    )
    out = []
    for i, (width, angle, ratio) in enumerate(specs):
        r = dict(
            step=i,
            item=0,
            depth=1,
            width=float(width),
            length=float(width / ratio),
            angle=float(np.deg2rad(angle)),
            center=rng.uniform(100, 156, 2).tolist(),
            sigma=float((0.8, 1.2, 1.8)[i % 3]),
            opacity=float((0.3, 0.4, 0.65)[i % 3]),
            angle_kind=2,
            canonical_style=False,
            square=ratio == 1.0,
            filled=False,
        )
        validate(r)
        out.append(r)
    return out


def arrays(split):
    recipes = records(split)
    rows = [render(r) for r in recipes]
    data = {k: np.stack([row[k] for row in rows]) for k in rows[0]}
    choice = np.asarray(preferred_original(data["direction"]))
    data["preferred_clean"] = np.where(
        choice[:, None, None, None], data["clean"], data["axis_clean"]
    )
    data["preferred_original"] = choice
    assert all(
        (a["source"] < 1).any() and (a["clean"] < 1).any() and (a["axis_clean"] < 1).any()
        for a in rows
    )
    return data, recipes
