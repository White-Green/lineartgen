"""Native-resolution spatial graph preview through the existing Krita BGRA API."""

import hashlib
import json
import operator
import os
import threading
from pathlib import Path

import numpy as np

from .curve_inputs import TILE, levels_from_bgra, tile_inputs
from .graph import render
from .images import PyramidPlan, prepare_input, to_bgra


class SpatialLineartModel:
    def __init__(self, manifest, num_threads=None, *, provider="auto", device_id=0):
        import onnxruntime as ort

        self._path = Path(manifest)
        self.metadata = json.loads(self._path.read_text())
        required = dict(
            format_version=1,
            representation="spatial_intent_graph",
            tile_size=TILE,
            core_size=128,
            input_channels=5,
            output_channels=9,
            normalization="separate_layers_sum",
            noise_rng="numpy_seedsequence_pcg64_v1",
            noise_after_downsample=True,
            renderer="attraction_graph_native_width_v1",
        )
        if any(self.metadata.get(k) != v for k, v in required.items()):
            raise ValueError("Unsupported spatial model manifest")
        entries = self.metadata.get("models", {})
        if not entries or any(str(k) not in {str(i) for i in range(1, 7)} for k in entries):
            raise ValueError("Spatial manifest must list supported recursion depths")
        self._models = {}
        for depth, entry in entries.items():
            filename = entry["file"]
            if Path(filename).name != filename:
                raise ValueError("Spatial model filenames must be local basenames")
            path = self._path.parent / filename
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"Spatial model digest mismatch: {filename}")
            self._models[int(depth)] = path
        if provider not in ("auto", "cpu", "directml"):
            raise ValueError("provider must be auto, cpu or directml")
        use_dml = provider != "cpu" and "DmlExecutionProvider" in ort.get_available_providers()
        if provider == "directml" and not use_dml:
            raise RuntimeError("DirectML is unavailable; install onnxruntime-directml")
        threads = (
            min(4, os.cpu_count() or 1) if num_threads is None else operator.index(num_threads)
        )
        if threads < 1:
            raise ValueError("num_threads must be positive")
        self._options = ort.SessionOptions()
        self._options.intra_op_num_threads = threads
        self._providers = ["CPUExecutionProvider"]
        if use_dml:
            self._options.enable_mem_pattern = False
            self._options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self._providers.insert(
                0, ("DmlExecutionProvider", {"device_id": operator.index(device_id)})
            )
        self.backend = "onnxruntime spatial graph"
        self.device = (
            "DmlExecutionProvider, CPUExecutionProvider" if use_dml else "CPUExecutionProvider"
        )
        self._sessions = {}
        self._lock = threading.Lock()

    def _session(self, depth):
        import onnxruntime as ort

        if depth not in self._sessions:
            session = ort.InferenceSession(
                str(self._models[depth]), sess_options=self._options, providers=self._providers
            )
            actual = {v.name: v.shape for v in session.get_inputs()}
            expected = {f"image_{i}": [1, TILE // 2**i, TILE // 2**i, 5] for i in range(depth)}
            if actual != expected or [v.name for v in session.get_outputs()] != ["graph"]:
                raise ValueError("Unexpected spatial ONNX input/output contract")
            self._sessions[depth] = session
        return self._sessions[depth]

    def infer(self, *, scribble, lineart, width, height, strength, seed, denoise_steps):
        width, height, seed, depth = map(operator.index, (width, height, seed, denoise_steps))
        if min(width, height) < 1:
            raise ValueError("Image dimensions must be positive")
        if depth not in self._models:
            raise ValueError(
                f"Unsupported recursion depth {depth}; available: {sorted(self._models)}"
            )
        if not np.isfinite(strength) or not 0 <= strength <= 1 or not 0 <= seed < 2**64:
            raise ValueError("Invalid strength or seed")
        plan = PyramidPlan((height, width), (height, width), ((height, width),))
        clean, noise = prepare_input(scribble, lineart, plan, strength, image_scale=1.0)
        if strength == 0 or not np.any(noise > 0):
            return to_bgra(clean, plan, alpha_threshold=0.0)
        levels, rows, cols = levels_from_bgra(
            scribble, lineart, width, height, strength, seed, depth
        )
        prediction = np.empty((1, height, width, 1), np.float32)
        with self._lock:
            session = self._session(depth)
            for y in range(0, height, rows.core):
                for x in range(0, width, cols.core):
                    inputs = tuple(image[..., :5] for image in tile_inputs(levels, y, x))
                    output = session.run(
                        None, {f"image_{i}": image for i, image in enumerate(inputs)}
                    )[0]
                    if output.shape != (1, TILE, TILE, 9) or not np.isfinite(output).all():
                        raise RuntimeError("Spatial ONNX inference produced an invalid graph")
                    raster = render(output[0], max_width=None)
                    h, w = min(rows.core, height - y), min(cols.core, width - x)
                    prediction[0, y : y + h, x : x + w] = raster[
                        rows.pad : rows.pad + h, cols.pad : cols.pad + w
                    ]
        prediction = np.where(noise > 0, prediction, clean)
        return to_bgra(prediction, plan, alpha_threshold=0.0)
