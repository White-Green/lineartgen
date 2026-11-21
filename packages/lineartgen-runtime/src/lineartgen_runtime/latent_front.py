"""Mask-conditioned sampling of a frozen line-art latent diffusion front end."""

import json
from pathlib import Path

import numpy as np

from .latent_context import StrokeContext, max_pool, repeat2, sigmoid
from .latent_gap import (
    agreed_role,
    completion_mask,
    facing_endpoints,
    paper_opacity,
    selected_coverage,
)
from .latent_noise import stage_noise


def halve(image):
    h, w, c = image.shape
    padded = np.pad(image, ((0, h % 2), (0, w % 2), (0, 0)), mode="edge")
    return padded.reshape((h + 1) // 2, 2, (w + 1) // 2, 2, c).mean((1, 3))


def pad_factor(image, factor):
    h, w = image.shape[:2]
    return np.pad(image, ((0, -h % factor), (0, -w % factor), (0, 0)), mode="edge")


class LatentFront:
    def __init__(self, folder, options, providers, tile_latents=64):
        import onnxruntime as ort

        folder = Path(folder)
        config = json.loads((folder / "latent.json").read_text(encoding="utf-8"))
        self.factor = config["factor"]
        self.has_readout = config.get("has_readout", True)
        self.steps = config["steps"]
        self.time_power = config.get("time_power", 1.0)
        self.eta = config.get("eta", 1.0)
        self.tile_latents = tile_latents
        self.slope = config["selection_slope"]
        self.role_offset = config.get("role_offset", 0.0)
        self.local_limit = config.get("local_limit", 0.0)
        self.direct_bridge = config.get("direct_bridge", False)
        self.core_transitions = config.get("core_transitions", False)
        self.free_completion = config.get("free_completion", False)
        self.supported_completion = config.get("supported_completion", False)
        self.directional_completion = config.get("directional_completion", False)
        self.endpoint_completion = config.get("endpoint_completion", False)
        self.contour_agreement = config.get("contour_agreement", False)
        self.finest_readout = config.get("finest_readout", False)
        self.sample_readout = config.get("sample_readout", False)
        self.separate_ink = config.get("separate_ink", False)
        self.coverage_decision = config.get("coverage_decision")
        if self.steps < 1 or not 0 <= self.eta <= 1 or self.time_power <= 0:
            raise ValueError("Invalid latent diffusion time schedule")
        with np.load(folder / "context.npz", allow_pickle=False) as w:
            self.weights = {k: w[k] for k in w.files}
        self.mean = self.weights["mean"].reshape(1, 1, -1)
        self.std = self.weights["std"].reshape(1, 1, -1)
        names = ["encoder", "decoder", "fine", "coarse"]
        if self.has_readout:
            names.append("stroke")
        if self.sample_readout:
            names.append("fusion")
        self.sessions = {
            name: ort.InferenceSession(
                str(folder / (name + ".onnx")), sess_options=options, providers=providers
            )
            for name in names
        }
        self.device = ", ".join(self.sessions["fine"].get_providers())

    def _forward(self, name, inputs):
        return [
            x[0]
            for x in self.sessions[name].run(
                None,
                {k: np.ascontiguousarray(v[None], dtype=np.float32) for k, v in inputs.items()},
            )
        ]

    def run(self, name, **inputs):
        # Local ONNX graphs run on overlapping tiles. Context and random arrays
        # always span the canvas, including components that cross tile boundaries.
        f = self.factor
        if name == "encoder":
            multipliers, scales, halo = {"x": f}, (1,), 4
        elif name == "decoder":
            multipliers, scales, halo = {"x": 1}, (f,), 4
        elif name == "fusion":
            multipliers, scales, halo = {"x": f}, (f,), 3
        elif name == "stroke":
            multipliers = {
                "rough": f,
                "previous": f,
                "proposal": f,
                "features": 1,
                "fields": f // 2,
            }
            scales, halo = (f, f // 2), 4
        elif name == "merge":
            multipliers, scales, halo = {"x": 1}, (1,), 18
        else:
            multipliers = dict.fromkeys(("state", "guide", "observation", "previous"), 1)
            scales, halo = (1, 1), 12
        ref = next(iter(multipliers))
        h, w = (size // multipliers[ref] for size in inputs[ref].shape[:2])
        tile = self.tile_latents * (f if name == "merge" else 1)
        if not tile or max(h, w) <= tile:
            return self._forward(name, inputs)
        result = []
        for y in range(0, h, tile):
            a, end_y, bottom = max(0, y - halo), min(h, y + tile), min(h, y + tile + halo)
            for x in range(0, w, tile):
                b, end_x, right = max(0, x - halo), min(w, x + tile), min(w, x + tile + halo)
                feeds = inputs.copy()
                for key, m in multipliers.items():
                    feeds[key] = inputs[key][a * m : bottom * m, b * m : right * m]
                outputs = self._forward(name, feeds)
                if not result:
                    result = [
                        np.empty((h * m, w * m, v.shape[-1]), v.dtype)
                        for m, v in zip(scales, outputs, strict=True)
                    ]
                for dest, m, value in zip(result, scales, outputs, strict=True):
                    dest[y * m : end_y * m, x * m : end_x * m] = value[
                        (y - a) * m : (end_y - a) * m, (x - b) * m : (end_x - b) * m
                    ]
        return result

    def encode(self, ink, fine):
        if not fine:
            return (ink - 0.15) / 0.35
        (mu,) = self.run("encoder", x=pad_factor(ink, self.factor))
        return (mu - self.mean) / self.std

    def decode(self, z, fine):
        if not fine:
            return np.clip(0.15 + 0.35 * z, 0, 1)
        (ink,) = self.run("decoder", x=z * self.std + self.mean)
        return ink

    def generate(self, raw, power, seed, depth):
        if raw.ndim != 2 or not 1 <= depth <= 6:
            raise ValueError("Expected a two-dimensional rough and one to six spatial levels")
        levels = [raw.astype(np.float32, copy=False)[..., None]]
        powers = [np.broadcast_to(np.asarray(power, dtype=np.float32), raw.shape)[..., None]]
        for _ in range(depth - 1):
            levels.append(halve(levels[-1]))
            powers.append(halve(powers[-1]))
        previous = None
        for level in reversed(range(depth)):
            raw, power = levels[level], powers[level]
            h, w = raw.shape[:2]
            fine = min(h, w) >= 64
            draw = stage_noise(
                seed,
                level,
                (1, h, w, 1),
                self.factor,
                self.mean.size,
                self.steps,
                stochastic=self.eta > 0,
            )
            guide = self.encode(raw, fine)
            obs = self.encode(np.clip(raw + 0.125 * power * draw.observation[0], 0, 1), fine)
            active = np.full((1, 1, 1), previous is not None, np.float32)
            hint = (
                np.zeros_like(guide)
                if previous is None
                else self.encode(repeat2(previous)[:h, :w], fine)
            )
            x = obs + 0.35 * draw.initial[0]
            condition = guide
            readout = self.has_readout and (not self.finest_readout or level == 0)
            if fine and (readout or self.direct_bridge):
                rough = self.decode(guide, True)
                if self.free_completion:
                    blank = 1 - np.clip(max_pool(rough, 9) / 0.075, 0, 1)
                    if self.finest_readout and self.supported_completion:
                        blank = 1 - np.clip(max_pool(rough, 3) / 0.075, 0, 1)
                context = StrokeContext(rough, pad_factor(power, self.factor), self.weights)
                if self.direct_bridge:
                    confidence = np.where(
                        context.observed, 0.10 + 0.90 * sigmoid(context.prior), 1.0
                    )
                    condition = self.encode(rough * repeat2(confidence), True)
                prior_hint = self.decode(hint, True) * active
                if self.supported_completion:
                    nearby = max_pool(rough, 33)
                    coarse = max_pool(prior_hint, 5)
                    blank *= np.clip(np.maximum(nearby, 0.4 * coarse) / 0.08, 0, 1)
            for i in range(self.steps):
                t = ((self.steps - i) / self.steps) ** self.time_power
                following = ((self.steps - i - 1) / self.steps) ** self.time_power
                time = np.full((1, 1, 1), t, np.float32)
                denoised, features = self.run(
                    "fine" if fine else "coarse",
                    state=x,
                    observation=obs,
                    guide=condition,
                    previous=hint,
                    time=time,
                    active=active,
                )
                if fine and readout and (not self.core_transitions or i + 1 == self.steps):
                    proposal = self.decode(denoised, True)
                    shape, local = self.run(
                        "stroke",
                        rough=rough,
                        proposal=proposal,
                        previous=prior_hint,
                        features=features,
                        fields=context.fields,
                        time=time,
                    )
                    logits = context.select(local, features, proposal, self.local_limit)
                    role = sigmoid(self.slope * (logits - self.role_offset))
                    if self.contour_agreement:
                        cell = halve(role)
                        evidence = np.concatenate((cell, (cell < 0.2).astype(np.float32)), -1)
                        for group, _ in context.regions:
                            means = group.means(evidence)[group.ids]
                            agreed = (
                                agreed_role(means[..., :1], means[..., 1:], np)
                                * group.mask[..., None]
                            )
                            role = np.maximum(role, repeat2(agreed))
                    ink = shape * role
                    confidence = 1.0
                    if self.directional_completion:

                        def peak3(values):
                            return max_pool(values[0], 3)[None]

                        mask = completion_mask(rough[None], ink[None], np, peak3)[0]
                        if self.endpoint_completion:
                            mask *= facing_endpoints(rough[None], np, peak3)[0]
                    if self.sample_readout:
                        fields = np.concatenate((rough, proposal, prior_hint, ink, shape, role), -1)
                        (learned,) = self.run("fusion", x=fields)
                        if self.separate_ink:
                            visible, proposal = np.split(learned, 2, axis=-1)
                            if self.coverage_decision:
                                low, high, _ = self.coverage_decision
                                ink = selected_coverage(ink, visible, low, high, np)
                            else:
                                ink = visible
                        elif self.directional_completion:
                            confidence = learned
                        else:
                            ink = learned
                    if self.directional_completion:
                        ink = np.maximum(ink, proposal * mask * confidence)
                        if self.coverage_decision:
                            ink = paper_opacity(
                                ink, max_pool(ink, 5), self.coverage_decision[2], np
                            )
                    elif self.free_completion and not self.sample_readout:
                        ink += blank * (proposal - ink)
                    denoised = self.encode(ink, True)
                if i + 1 == self.steps:
                    x = denoised
                else:
                    ratio = np.float32(following / t)
                    scale = np.float32(0.35 * np.sqrt(following * (1 - ratio)))
                    if self.eta == 1:
                        x = ratio * x + (1 - ratio) * denoised + scale * draw.reverse[i, 0]
                    else:
                        center = denoised + t * (obs - denoised)
                        next_center = denoised + following * (obs - denoised)
                        carry = np.float32(np.sqrt(ratio * (1 - self.eta**2 * (1 - ratio))))
                        x = next_center + carry * (x - center)
                        if self.eta > 0:
                            x += self.eta * scale * draw.reverse[i, 0]
            previous = self.decode(x, fine)[:h, :w]
        return previous[..., 0]
