"""Bound CNN activation memory while preserving whole-image recursive feedback."""

import numpy as np


class TiledSession:
    def __init__(self, session, core=512, halo=64):
        if core < 8 or core % 8 or halo < 64 or halo % 8:
            raise ValueError(
                "CNN cores and halos must align to eight; halo must cover the receptive field"
            )
        self.session, self.core, self.halo = session, core, halo
        self.last_calls = []

    def run(self, output_names, inputs):
        if output_names != ["graph"] or set(inputs) != {"features"}:
            raise ValueError("Unexpected tiled stage call")
        features = inputs["features"]
        _, height, width, channels = features.shape
        assert features.shape[0] == 1 and channels == 10
        self.last_calls = []
        # Odd/coarsest shapes use their original Resize ratios. Fine image levels
        # from the BGRA preparation are aligned; their down/up ratios are exactly two.
        if max(height, width) <= self.core or height % 8 or width % 8:
            self.last_calls.append([height, width])
            return self.session.run(output_names, inputs)
        result = np.empty((1, height, width, 9), np.float32)
        for y in range(0, height, self.core):
            y1 = min(height, y + self.core)
            top, bottom = max(0, y - self.halo), min(height, y1 + self.halo)
            for x in range(0, width, self.core):
                x1 = min(width, x + self.core)
                left, right = max(0, x - self.halo), min(width, x1 + self.halo)
                self.last_calls.append([bottom - top, right - left])
                crop = np.ascontiguousarray(features[:, top:bottom, left:right])
                tile = self.session.run(["graph"], {"features": crop})[0]
                if tile.shape != (1, bottom - top, right - left, 9) or not np.isfinite(tile).all():
                    raise RuntimeError("Invalid tiled stage graph")
                result[:, y:y1, x:x1] = tile[:, y - top : y1 - top, x - left : x1 - left]
        return [result]
