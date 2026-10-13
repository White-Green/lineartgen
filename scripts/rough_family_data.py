"""Rough inputs with varying opacity, blur and smooth local displacements."""

import numpy as np
from scipy import ndimage

BASE = dict(opacity=0.4, sigma=1.2, amplitude=0.0, phase_x=0.0, phase_y=0.0, period=96.0)
STRESS = {
    "base": BASE,
    "faint": BASE | dict(opacity=0.2),
    "dark": BASE | dict(opacity=0.8),
    "sharp": BASE | dict(sigma=0.6),
    "soft": BASE | dict(sigma=2.4),
    "rough": BASE | dict(opacity=0.3, sigma=1.8, amplitude=1.5),
}


def augmentation(seed, step, item):
    rng = np.random.default_rng([seed, step, item, 0x415547])
    if rng.random() < 0.25:
        return BASE.copy()
    return dict(
        opacity=float(rng.uniform(0.2, 0.8)),
        sigma=float(rng.uniform(0.6, 2.4)),
        amplitude=float(rng.uniform(0, 1.5)),
        phase_x=float(rng.uniform(0, 2 * np.pi)),
        phase_y=float(rng.uniform(0, 2 * np.pi)),
        period=float(rng.uniform(64, 160)),
    )


def stress_options(name, index):
    result = STRESS[name].copy()
    if name == "rough":
        rng = np.random.default_rng([2026101318, index])
        result |= dict(
            phase_x=float(rng.uniform(0, 2 * np.pi)), phase_y=float(rng.uniform(0, 2 * np.pi))
        )
    return result


def rough_source(mixture, options):
    alpha = (1 - mixture[..., 0]) / 2
    if options["amplitude"]:
        yy, xx = np.indices(alpha.shape, dtype=np.float32)
        frequency = 2 * np.pi / options["period"]
        # Smooth, bounded local shifts of the rough drawing, not its acceptable clean family.
        x = xx + options["amplitude"] * np.sin(yy * frequency + options["phase_x"])
        y = yy + options["amplitude"] * np.sin(xx * frequency + options["phase_y"])
        alpha = ndimage.map_coordinates(alpha, [y, x], order=1, mode="constant", cval=0)
    alpha = ndimage.gaussian_filter(alpha, options["sigma"], mode="constant", cval=0)
    return (1 - 2 * options["opacity"] * alpha)[..., None].astype(np.float32)
