"""JAX-free graph inference and opaque stroke rendering for the Krita preview."""

import hashlib
import json
import operator
import os
import threading
from pathlib import Path

import numpy as np

from .global_normalization import normalized_global_rough
from .images import PyramidPlan, prepare_input, to_bgra
from .layer_normalization import bgra_layer_components, normalized_layer_guide
from .masked_normalization import normalized_masked_guide
from .painted_normalization import normalized_painted_guide

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


def decode_segments(head, radius=1.0, maximum=2.0, tolerance=1e-5, *, max_width=8):
    h, w, channels = head.shape
    if channels != 9 or not np.isfinite(head).all():
        raise ValueError("Expected finite graph predictions with nine channels")
    occupied = head[..., 0] >= 0
    yy, xx = np.mgrid[:h, :w]
    xy = np.stack((xx + 0.5, yy + 0.5), -1) + radius * np.tanh(head[..., 1:3])
    limit = 2 * max(h, w) if max_width is None else float(max_width)
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("max_width must be positive and finite")
    widths = np.exp(np.clip(head[..., 3], -8 if max_width == 8 else -16, np.log(limit)))
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
    # Wide native strokes have much larger bounding boxes than the legacy
    # 8px strokes. Bound temporary raster arrays as well as the canvas tile.
    extent = min(max(h, w) * s, int(np.ceil(widths.max() * s)) + 4 * s)
    batch_size = max(1, min(128, 2_000_000 // max(1, extent**2)))
    for start in range(0, len(segments), batch_size):
        lines = segments[start : start + batch_size]
        ww = np.asarray(widths[start : start + batch_size], np.float64)
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


def render(head, supersampling=4, *, tile_size=256, max_width=8):
    """Opaque capsule union, with bounded supersampling memory on large canvases."""
    h, w, _ = head.shape
    segments, widths = decode_segments(head, max_width=max_width)
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
            self.metadata.get("format_version") not in (2, 3, 4, 5, 6, 7, 8, 9)
            or self.metadata.get("representation") != "attraction_graph"
            or self.metadata.get("levels") != 6
        ):
            raise ValueError("Unsupported graph model manifest")
        if self.metadata["format_version"] >= 3 and "max_stroke_width" not in self.metadata:
            raise ValueError("Native graph manifest requires an explicit max_stroke_width policy")
        if self.metadata["format_version"] >= 4 and "guide_normalization" not in self.metadata:
            raise ValueError("Graph manifest v4 requires an explicit guide_normalization policy")
        policy = self.metadata.get("guide_normalization", "local_peak_full")
        if policy not in (
            "local_peak_full",
            "protected_peak_full",
            "painted_peak_full",
            "separate_layers_sum",
            "separate_layers_max",
        ):
            raise ValueError("Unsupported graph guide_normalization policy")
        if policy != "local_peak_full" and self.metadata["format_version"] < 4:
            raise ValueError("Protected normalization requires graph manifest v4")
        detail_input = self.metadata.get("detail_input")
        if self.metadata["format_version"] >= 5:
            if detail_input not in ("guide", "composite", "painted"):
                raise ValueError("Graph manifest v5 requires a valid detail_input")
        elif detail_input is not None:
            raise ValueError("An explicit detail input requires graph manifest v5")
        gate = self.metadata.get("width_gate")
        if self.metadata["format_version"] == 6:
            if (
                not isinstance(gate, dict)
                or gate.get("kind") != "independent_painted_width_v1"
                or gate.get("file") != "width-gate.onnx"
                or detail_input != "painted"
                or policy != "separate_layers_sum"
            ):
                raise ValueError("Invalid independent width gate contract")
        elif gate is not None:
            raise ValueError("A width gate requires graph manifest v6")
        context_input = self.metadata.get("context_input")
        if self.metadata["format_version"] in (7, 9):
            if (
                context_input not in ("control", "painted", "raw")
                or detail_input != "painted"
                or policy != "separate_layers_sum"
            ):
                raise ValueError("Invalid rough and painted context contract")
        elif context_input is not None:
            raise ValueError("Separate context inputs require graph manifest v7")
        rough_normalization = self.metadata.get("rough_input_normalization")
        if self.metadata["format_version"] == 9:
            if rough_normalization not in ("identity", "global_peak") or context_input != "raw":
                raise ValueError("Invalid original rough normalization contract")
        elif rough_normalization is not None:
            raise ValueError("Original rough normalization requires graph manifest v9")
        width_feedback = self.metadata.get("width_feedback")
        if self.metadata["format_version"] == 8:
            if (
                width_feedback not in ("control", "width")
                or detail_input != "painted"
                or policy != "separate_layers_sum"
            ):
                raise ValueError("Invalid recursive width feedback contract")
        elif width_feedback is not None:
            raise ValueError("Recursive width feedback requires graph manifest v8")
        if provider not in ("auto", "cpu", "directml"):
            raise ValueError("provider must be auto, cpu or directml")
        threads = (
            min(4, os.cpu_count() or 1) if num_threads is None else operator.index(num_threads)
        )
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
            expected_inputs = {"guide", "noise_level", "insert"}
            if detail_input:
                expected_inputs.add("detail")
            if context_input:
                expected_inputs.add("rough")
            if names != expected_inputs:
                raise ValueError("Unexpected graph input contract")
            if width_feedback:
                insert = next(x for x in session.get_inputs() if x.name == "insert")
                if insert.shape[-1] != 3:
                    raise ValueError("Recursive width feedback requires three insert channels")
            self.sessions.append(session)
        self.width_gate = None
        if gate:
            filename = gate["file"]
            payload = (path.parent / filename).read_bytes()
            if hashlib.sha256(payload).hexdigest() != self.metadata["files"].get(filename):
                raise ValueError(f"Graph model checksum mismatch: {filename}")
            self.width_gate = ort.InferenceSession(
                payload, sess_options=options, providers=providers
            )
            if [x.name for x in self.width_gate.get_inputs()] != ["features"] or [
                x.name for x in self.width_gate.get_outputs()
            ] != ["weight"]:
                raise ValueError("Unexpected width gate input/output contract")
        self.backend = "onnxruntime graph"
        self.device = ", ".join(self.sessions[0].get_providers())
        self._lock = threading.Lock()

    def stage(self, guide, noise, insert, level, *, tile_size=512, detail=None, rough=None):
        """Overlap exceeds the full network receptive radius; coarse pooling stays aligned."""
        session = self.sessions[level]

        def run(g, n, i, d, r):
            feeds = dict(guide=g, noise_level=n, insert=i)
            if d is not None:
                feeds["detail"] = d
            if r is not None:
                feeds["rough"] = r
            return session.run(["head"], feeds)[0]

        h, w = guide.shape[1:3]
        if tile_size is None or max(h, w) <= tile_size:
            return run(guide, noise, insert, detail, rough)
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
                    detail[section] if detail is not None else None,
                    rough[section] if rough is not None else None,
                )
                result[:, y:y1, x:x1] = tile[:, y - top : y1 - top, x - left : x1 - left]
        return result

    def heads(self, source, noise, depth, *, layer_guide=None, detail_guide=None, rough_guide=None):
        """Normalize once, then construct the area-mean pyramid (padded NHWC)."""
        if not 1 <= depth <= 6:
            raise ValueError("Graph recursion depth must be between 1 and 6")
        if source.shape != noise.shape or source.ndim != 4 or source.shape[0] != 1:
            raise ValueError("Expected matching NHWC source and strength")
        if source.shape[-1] != 1 or any(n % 64 for n in source.shape[1:3]):
            raise ValueError("Graph spatial dimensions must be multiples of 64")
        policy = self.metadata.get("guide_normalization", "local_peak_full")
        if policy in ("separate_layers_sum", "separate_layers_max"):
            if layer_guide is None:
                raise ValueError("This policy requires the original layer guide")
            if layer_guide.shape != source.shape or not np.isfinite(layer_guide).all():
                raise ValueError("Invalid original layer guide")
            guide = layer_guide
        elif policy == "painted_peak_full":
            guide = normalized_painted_guide(source, noise)
        elif policy == "protected_peak_full":
            guide = normalized_masked_guide(source, noise)
        else:
            guide = normalized_guide(source)
        detail_input = self.metadata.get("detail_input")
        if detail_input == "guide":
            detail_guide = guide
        elif detail_input == "composite":
            detail_guide = 1 - 0.4 * (1 - source)
        elif detail_input == "painted" and detail_guide is None:
            raise ValueError("Painted detail requires the original layer coverage")
        if detail_guide is not None and (
            detail_guide.shape != source.shape or not np.isfinite(detail_guide).all()
        ):
            raise ValueError("Invalid detail guide")
        if self.metadata.get("context_input") and rough_guide is None:
            raise ValueError("Separate context requires the original rough intensity")
        if rough_guide is not None and (
            rough_guide.shape != source.shape or not np.isfinite(rough_guide).all()
        ):
            raise ValueError("Invalid original rough guide")
        if self.metadata.get("rough_input_normalization") == "global_peak":
            rough_guide = normalized_global_rough(rough_guide)
        guides, levels = [guide], [noise]
        details = None if detail_guide is None else [detail_guide]
        roughs = None if rough_guide is None else [rough_guide]
        for _ in range(depth - 1):
            guides.append(downsample(guides[-1]))
            levels.append(downsample(levels[-1]))
            if details is not None:
                details.append(downsample(details[-1]))
            if roughs is not None:
                roughs.append(downsample(roughs[-1]))
        g = guides[-1]
        channels = 3 if self.metadata.get("width_feedback") else 1
        insert = np.zeros((1, g.shape[1] // 2, g.shape[2] // 2, channels), np.float32)
        result = [None] * depth
        for level in reversed(range(depth)):
            extra = {}
            if details is not None:
                extra["detail"] = details[level]
            if roughs is not None:
                extra["rough"] = roughs[level]
            head = self.stage(guides[level], levels[level], insert, level, **extra)
            if not np.isfinite(head).all():
                raise RuntimeError("Graph inference produced nonfinite output")
            result[level] = head
            v = 1 / (1 + np.exp(-np.clip(head[..., :1], -80, 80)))
            fill = 1 / (1 + np.exp(-np.clip(head[..., 8:9], -80, 80)))
            insert = 1 - 2 * (1 - (1 - v) * (1 - fill))
            if self.metadata.get("width_feedback"):
                next_logwidth = np.clip(head[..., 3:4] + np.float32(np.log(2)), -8, 8) / 4
                insert = np.concatenate((insert, v * next_logwidth, v), axis=-1)
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
        layer_guide = None
        detail_guide = None
        rough_guide = None
        policy = self.metadata.get("guide_normalization", "local_peak_full")
        if policy in ("separate_layers_sum", "separate_layers_max"):
            rough, coverage, alpha = bgra_layer_components(scribble, lineart, height, width)
            padded_rough, padded_coverage, padded_alpha = (
                np.ones_like(source),
                np.zeros_like(source),
                np.zeros_like(source),
            )
            padded_rough[crop], padded_coverage[crop], padded_alpha[crop] = rough, coverage, alpha
            layer_guide = normalized_layer_guide(
                padded_rough, padded_coverage, padded_alpha, policy.removeprefix("separate_layers_")
            )
            if self.metadata.get("detail_input") == "painted":
                detail_guide = 1 - 0.8 * padded_coverage
            if self.metadata.get("context_input"):
                rough_guide = 1 - 0.4 * (1 - padded_rough)
        elif self.metadata.get("detail_input") == "painted":
            _, coverage, _ = bgra_layer_components(scribble, lineart, height, width)
            detail_guide = np.ones_like(source)
            detail_guide[crop] = 1 - 0.8 * coverage
        with self._lock:
            extra = {"rough_guide": rough_guide} if rough_guide is not None else {}
            head = (
                self.heads(source, levels, depth)
                if layer_guide is None and detail_guide is None
                else self.heads(
                    source,
                    levels,
                    depth,
                    layer_guide=layer_guide,
                    detail_guide=detail_guide,
                    **extra,
                )
            )[0][0]
            if getattr(self, "width_gate", None) is not None and np.any(detail_guide < 1):
                from .width_gate import fuse_width

                painted_head = self.heads(
                    np.ones_like(source),
                    levels,
                    depth,
                    layer_guide=detail_guide,
                    detail_guide=detail_guide,
                )[0]
                head = fuse_width(self.width_gate, head[None], painted_head, detail_guide, levels)[
                    0
                ]
            prediction = render(
                head,
                self.metadata["supersampling"],
                max_width=self.metadata.get("max_stroke_width", 8),
            )[None][crop]
        prediction = np.where(noise > 0, prediction, clean)
        return to_bgra(prediction, plain, alpha_threshold=0.0)
