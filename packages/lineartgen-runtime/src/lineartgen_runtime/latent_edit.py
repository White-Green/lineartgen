"""Pixel-space editing around a one-channel latent line-art proposal."""

import numpy as np

from .images import PyramidPlan, prepare_input, to_bgra
from .layer_normalization import bgra_layer_components


class PixelEdit:
    def __init__(self, scribble, lineart, height, width, strength):
        self.plan = PyramidPlan((height, width), (height, width), ((height, width),))
        self.clean, noise = prepare_input(scribble, lineart, self.plan, strength, image_scale=1.0)
        rough, coverage, alpha = bgra_layer_components(scribble, lineart, height, width)
        self.rough = (1 - rough[0, ..., 0]) * 0.5
        self.coverage = coverage[0, ..., 0]
        self.alpha = alpha[0, ..., 0]
        self.power = noise[0, ..., 0]

    def guide(self, painted=True):
        if painted:
            return self.coverage + (1 - self.alpha) * self.rough
        return self.rough

    def features(self, proposal):
        return np.stack((proposal, self.rough, self.coverage, self.alpha, self.power), -1)

    def render(self, underpaint=None):
        if underpaint is None:
            return to_bgra(self.clean, self.plan, alpha_threshold=0)
        ink = self.coverage + (1 - self.alpha) * np.clip(underpaint, 0, 1)
        gray = np.where(self.power > 0, 1 - 2 * ink, self.clean[0, ..., 0])
        return to_bgra(gray[None, ..., None], self.plan, alpha_threshold=0)
