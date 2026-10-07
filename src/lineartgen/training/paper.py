"""Localized teacher-paper errors, including isolated ink far from the true line."""

import jax
import jax.numpy as jnp


def paper_peak_loss(prediction, target, active, *, tile=8):
    """Average the worst excess-opacity error per nonoverlapping editable paper tile.

    Teacher ink (> .02 opacity) and protected pixels cannot supply pool errors.
    Partial edge tiles and partially protected tiles count once if any paper is
    editable. Empty images contribute zero. The continuous teacher is stationary;
    this does not impose binary targets or claim topology preservation.
    """
    truth = jax.lax.stop_gradient(jnp.clip((1 - target) / 2, 0, 1))
    paper = active & (truth <= 0.02)
    ink = jnp.clip((1 - prediction) / 2, 0, 1)
    excess = jnp.where(paper, jax.nn.relu(ink - truth) ** 2, 0.0)
    window = (1, tile, tile, 1)

    def pool(values):
        return jax.lax.reduce_window(values, -jnp.inf, jax.lax.max, window, window, "SAME")

    # SAME padding is excluded from both error and eligible-tile counts.
    maxima = pool(excess)
    eligible = pool(paper.astype(excess.dtype))
    axes = (1, 2, 3)
    return ((maxima * eligible).sum(axes) / jnp.maximum(eligible.sum(axes), 1)).mean()
