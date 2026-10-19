"""JAX-free graph inference and opaque stroke rendering for the Krita preview."""

import hashlib
import json
import operator
import threading
from pathlib import Path

import numpy as np

from .images import PyramidPlan, prepare_input, to_bgra

DIRECTIONS = ((1, 0), (0, 1), (1, 1), (-1, 1))


def downsample(image):
    b, h, w, c = image.shape
    return image.reshape(b, h // 2, 2, w // 2, 2, c).mean((2, 4))


def normalized_guide(source):
    ink = np.clip((1 - source) / 2, 0, 1)
    peak = ink
    for axis in (1, 2):
        pads = [(0, 0)] * 4
        pads[axis] = (16, 16)
        padded = np.pad(peak, pads)
        peak = np.zeros_like(peak)
        for shift in range(33):
            slices = [slice(None)] * 4
            slices[axis] = slice(shift, shift + peak.shape[axis])
            np.maximum(peak, padded[tuple(slices)], out=peak)
    return 1 - 0.8 * ink / np.maximum(peak, 0.02)


def decode_segments(head, radius=1.0, maximum=2.0, tolerance=1e-5):
    h, w, channels = head.shape
    if channels != 9 or not np.isfinite(head).all():
        raise ValueError("Expected finite graph predictions with nine channels")
    occupied = head[..., 0] >= 0
    yy, xx = np.mgrid[:h, :w]
    xy = np.stack((xx + 0.5, yy + 0.5), -1) + radius * np.tanh(head[..., 1:3])
    widths = np.exp(np.clip(head[..., 3], -8, np.log(8)))
    degree = np.zeros((h, w), np.int32)
    segments, thicknesses = [], []
    for k, (dx, dy) in enumerate(DIRECTIONS):
        y0, y1 = max(0, -dy), min(h, h - dy)
        x0, x1 = max(0, -dx), min(w, w - dx)
        a = np.s_[y0:y1, x0:x1]
        b = np.s_[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        valid = occupied[a] & occupied[b] & (head[..., 4 + k][a] >= 0)
        valid &= np.linalg.norm(xy[a] - xy[b], axis=-1) <= maximum + tolerance
        segments.append(np.stack((xy[a][valid], xy[b][valid]), axis=1))
        thicknesses.append(np.stack((widths[a][valid], widths[b][valid]), axis=1))
        degree[a] += valid
        degree[b] += valid
    isolated = occupied & (degree == 0)
    segments.append(np.repeat(xy[isolated, None, :], 2, axis=1))
    thicknesses.append(np.repeat(widths[isolated, None], 2, axis=1))
    return np.concatenate(segments), np.concatenate(thicknesses)


def _render_tile(segments, widths, origin, size, supersampling):
    x0, y0 = origin
    h, w = size
    s = supersampling
    alpha = np.zeros((h * s, w * s), np.float64)
    floor = np.array([x0, y0]) * s
    ceiling = floor + np.array([w, h]) * s
    for start in range(0, len(segments), 128):
        lines = segments[start : start + 128]
        ww = np.asarray(widths[start : start + 128], np.float64)
        pad = ww.max(axis=1) / 2 + 1 / s
        low = np.maximum(np.floor((lines.min(1) - pad[:, None]) * s).astype(int), floor)
        high = np.minimum(np.ceil((lines.max(1) + pad[:, None]) * s).astype(int), ceiling)
        valid = np.all(high > low, axis=1)
        lines, ww, low, high = lines[valid], ww[valid], low[valid], high[valid]
        if not len(lines):
            continue
        width, height = (high - low).max(axis=0)
        xx = low[:, 0, None, None] + np.arange(width)[None, None, :]
        yy = low[:, 1, None, None] + np.arange(height)[None, :, None]
        dx = (xx + 0.5) / s - lines[:, 0, 0, None, None]
        dy = (yy + 0.5) / s - lines[:, 0, 1, None, None]
        v = lines[:, 1] - lines[:, 0]
        vx, vy = v[:, 0, None, None], v[:, 1, None, None]
        denominator = np.maximum(np.sum(v * v, axis=1), 1e-12)[:, None, None]
        t = np.clip((dx * vx + dy * vy) / denominator, 0, 1)
        # Use the same operation order as the research renderer.
        px = lines[:, 0, 0, None, None] + t * vx
        py = lines[:, 0, 1, None, None] + t * vy
        distance = np.sqrt(((xx + 0.5) / s - px) ** 2 + ((yy + 0.5) / s - py) ** 2 + 1e-12)
        radius = 0.5 * (ww[:, 0, None, None] + t * (ww[:, 1] - ww[:, 0])[:, None, None])
        coverage = np.where(radius > 0, np.clip(0.5 - s * (distance - radius), 0, 1), 0)
        active = (xx < high[:, 0, None, None]) & (yy < high[:, 1, None, None])
        indices = (yy - floor[1]) * (w * s) + (xx - floor[0])
        np.maximum.at(alpha.ravel(), indices[active], coverage[active])
    return (1 - 2 * alpha.reshape(h, s, w, s).mean((1, 3))).astype(np.float32)[..., None]


def render(head, supersampling=4, *, tile_size=256):
    """Opaque capsule union, with bounded supersampling memory on large canvases."""
    h, w, _ = head.shape
    segments, widths = decode_segments(head)
    pad = widths.max(axis=1) / 2 + 1 / supersampling
    low = segments.min(axis=1) - pad[:, None]
    high = segments.max(axis=1) + pad[:, None]
    image = np.ones((h, w, 1), np.float32)
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            height, width = min(tile_size, h - y), min(tile_size, w - x)
            hit = (
                (low[:, 0] < x + width)
                & (high[:, 0] > x)
                & (low[:, 1] < y + height)
                & (high[:, 1] > y)
            )
            if np.any(hit):
                image[y : y + height, x : x + width] = _render_tile(
                    segments[hit], widths[hit], (x, y), (height, width), supersampling
                )
    image[head[..., 8] >= 0] = -1
    return image


class GraphLineartModel:
    def __init__(self, manifest, num_threads=None, *, provider="auto", device_id=0):
        import onnxruntime as ort

        path = Path(manifest)
        self.metadata = json.loads(path.read_text())
        if (
            self.metadata.get("format_version") != 2
            or self.metadata.get("representation") != "attraction_graph"
            or self.metadata.get("levels") != 6
        ):
            raise ValueError("Unsupported graph model manifest")
        if provider not in ("auto", "cpu", "directml"):
            raise ValueError("provider must be auto, cpu or directml")
        threads = 1 if num_threads is None else operator.index(num_threads)
        if threads < 1:
            raise ValueError("num_threads must be positive")
        use_dml = provider != "cpu" and "DmlExecutionProvider" in ort.get_available_providers()
        if provider == "directml" and not use_dml:
            raise RuntimeError("DirectML is unavailable; install onnxruntime-directml")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        providers = ["CPUExecutionProvider"]
        if use_dml:
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            providers.insert(0, ("DmlExecutionProvider", {"device_id": device_id}))
        self.sessions = []
        for level in range(6):
            filename = f"graph-level-{level}.onnx"
            payload = (path.parent / filename).read_bytes()
            if hashlib.sha256(payload).hexdigest() != self.metadata["files"][filename]:
                raise ValueError(f"Graph model checksum mismatch: {filename}")
            session = ort.InferenceSession(payload, sess_options=options, providers=providers)
            names = {x.name for x in session.get_inputs()}
            if names != {"guide", "noise_level", "insert"}:
                raise ValueError("Unexpected graph input contract")
            self.sessions.append(session)
        self.backend = "onnxruntime graph"
        self.device = ", ".join(self.sessions[0].get_providers())
        self._lock = threading.Lock()

    def stage(self, guide, noise, insert, level, *, tile_size=512):
        """Overlap exceeds the full network receptive radius; coarse pooling stays aligned."""
        session = self.sessions[level]

        def run(g, n, i):
            return session.run(["head"], dict(guide=g, noise_level=n, insert=i))[0]

        h, w = guide.shape[1:3]
        if tile_size is None or max(h, w) <= tile_size:
            return run(guide, noise, insert)
        if tile_size % 32:
            raise ValueError("Stage tile size must be a multiple of 32")
        # At level zero, the widest branch reaches 140 pixels. 192 also
        # covers bilinear interpolation and aligns every coarse pooling grid.
        halo = 192
        result = np.empty((1, h, w, 9), np.float32)
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                y1, x1 = min(h, y + tile_size), min(w, x + tile_size)
                top, left = max(0, y - halo), max(0, x - halo)
                bottom, right = min(h, y1 + halo), min(w, x1 + halo)
                section = np.s_[:, top:bottom, left:right, :]
                tile = run(
                    guide[section],
                    noise[section],
                    insert[:, top // 2 : bottom // 2, left // 2 : right // 2],
                )
                result[:, y:y1, x:x1] = tile[:, y - top : y1 - top, x - left : x1 - left]
        return result

    def heads(self, source, noise, depth):
        """Normalize once, then construct the area-mean pyramid (padded NHWC)."""
        if not 1 <= depth <= 6:
            raise ValueError("Graph recursion depth must be between 1 and 6")
        if source.shape != noise.shape or source.ndim != 4 or source.shape[0] != 1:
            raise ValueError("Expected matching NHWC source and strength")
        if source.shape[-1] != 1 or any(n % 64 for n in source.shape[1:3]):
            raise ValueError("Graph spatial dimensions must be multiples of 64")
        guides, levels = [normalized_guide(source)], [noise]
        for _ in range(depth - 1):
            guides.append(downsample(guides[-1]))
            levels.append(downsample(levels[-1]))
        g = guides[-1]
        insert = np.zeros((1, g.shape[1] // 2, g.shape[2] // 2, 1), np.float32)
        result = [None] * depth
        for level in reversed(range(depth)):
            head = self.stage(guides[level], levels[level], insert, level)
            if not np.isfinite(head).all():
                raise RuntimeError("Graph inference produced nonfinite output")
            result[level] = head
            v = 1 / (1 + np.exp(-np.clip(head[..., :1], -80, 80)))
            fill = 1 / (1 + np.exp(-np.clip(head[..., 8:9], -80, 80)))
            insert = 1 - 2 * (1 - (1 - v) * (1 - fill))
        return tuple(result)

    def infer(self, *, scribble, lineart, width, height, strength, seed, denoise_steps):
        width, height = operator.index(width), operator.index(height)
        seed, depth = operator.index(seed), operator.index(denoise_steps)
        if min(width, height) < 16:
            raise ValueError("Both image dimensions must be at least 16")
        if not np.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("strength must be finite and between 0 and 1")
        if not 1 <= depth <= 6:
            raise ValueError("Lineart passes must be between 1 and 6")
        if not 0 <= seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        plain = PyramidPlan((height, width), (height, width), ((height, width),))
        clean, noise = prepare_input(scribble, lineart, plain, strength, image_scale=1.0)
        if strength == 0:
            return to_bgra(clean, plain, alpha_threshold=0.0)
        halo = self.metadata["halo"]
        shape = tuple((n + 2 * halo + 63) // 64 * 64 for n in (height, width))
        source = np.ones((1, *shape, 1), np.float32)
        levels = np.zeros_like(source)
        crop = np.s_[:, halo : halo + height, halo : halo + width, :]
        source[crop] = clean
        levels[crop] = noise
        if np.all(noise > 0):
            levels.fill(strength)
            levels[crop] = noise
        with self._lock:
            head = self.heads(source, levels, depth)[0][0]
            prediction = render(head, self.metadata["supersampling"])[None][crop]
        prediction = np.where(noise > 0, prediction, clean)
        return to_bgra(prediction, plain, alpha_threshold=0.0)
