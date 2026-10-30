"""Whole-image recursive ONNX inference with bounded CNN tiles and the Krita BGRA API."""

import hashlib
import json
import operator
import os
import threading
from pathlib import Path

import numpy as np
from .stage_feedback import forward
from .stage_tiles import TiledSession
from .curve_inputs import levels_from_bgra
from .graph import render
from .images import PyramidPlan, prepare_input, to_bgra


class StageLineartModel:
    def __init__(self, manifest, num_threads=None, *, provider="auto", device_id=0):
        import onnxruntime as ort

        self._path = Path(manifest)
        self.metadata = json.loads(self._path.read_text())
        expected = dict(
            format_version=1,
            representation="spatial_intent_stage_graph",
            input_channels=10,
            output_channels=9,
            normalization="separate_layers_sum",
            noise_rng="numpy_seedsequence_pcg64_v1",
            noise_after_downsample=True,
            global_recursion=True,
            minimum_stage_size=8,
            feedback="spatial_occupancy_fill_width_v1",
            renderer="attraction_graph_native_width_v1",
            depths=list(range(1, 7)),
        )
        if any(self.metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("Unsupported spatial stage manifest")
        entry = self.metadata["stage"]
        filename = entry["file"]
        if Path(filename).name != filename:
            raise ValueError("Stage model must be a local basename")
        self._model_path = self._path.parent / filename
        if hashlib.sha256(self._model_path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Stage model digest mismatch")
        if provider not in ("auto", "cpu", "directml"):
            raise ValueError("provider must be auto, cpu or directml")
        use_dml = provider != "cpu" and "DmlExecutionProvider" in ort.get_available_providers()
        if provider == "directml" and not use_dml:
            raise RuntimeError("DirectML is unavailable; install onnxruntime-directml")
        threads = min(4, os.cpu_count() or 1) if num_threads is None else operator.index(num_threads)
        if threads < 1:
            raise ValueError("num_threads must be positive")
        self._options = ort.SessionOptions()
        self._options.intra_op_num_threads = threads
        self._providers = ["CPUExecutionProvider"]
        if use_dml:
            self._options.enable_mem_pattern = False
            self._options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self._providers.insert(0, ("DmlExecutionProvider", {"device_id": operator.index(device_id)}))
        self.backend = "onnxruntime spatial stage graph"
        self.device = "not initialized"
        self._session = None
        self._lock = threading.Lock()
        self._get_session()

    def _get_session(self):
        import onnxruntime as ort

        if self._session is None:
            session = ort.InferenceSession(str(self._model_path), sess_options=self._options, providers=self._providers)
            inputs = {value.name: value.shape for value in session.get_inputs()}
            outputs = {value.name: value.shape for value in session.get_outputs()}
            if inputs != {"features": [1, "height", "width", 10]} or outputs != {"graph": [1, "height", "width", 9]}:
                raise ValueError("Unexpected spatial stage ONNX input/output contract")
            self.device = ", ".join(session.get_providers())
            self._session = TiledSession(session)
        return self._session

    def infer(self, *, scribble, lineart, width, height, strength, seed, denoise_steps):
        width, height, seed, depth = map(operator.index, (width, height, seed, denoise_steps))
        if min(width, height) < 1:
            raise ValueError("Image dimensions must be positive")
        if not 1 <= depth <= 6:
            raise ValueError("Recursion depth must be one to six")
        if not np.isfinite(strength) or not 0 <= strength <= 1 or not 0 <= seed < 2**64:
            raise ValueError("Invalid strength or seed")
        plan = PyramidPlan((height, width), (height, width), ((height, width),))
        clean, noise = prepare_input(scribble, lineart, plan, strength, image_scale=1.0)
        if strength == 0 or not np.any(noise > 0):
            return to_bgra(clean, plan, alpha_threshold=0.0)
        inputs, rows, cols = levels_from_bgra(scribble, lineart, width, height, strength, seed, depth)
        with self._lock:
            head = forward(self._get_session(), inputs)[0][0]
            prediction = render(head, max_width=None)[rows.pad:rows.pad + height, cols.pad:cols.pad + width]
        prediction = np.where(noise > 0, prediction[None], clean)
        return to_bgra(prediction, plan, alpha_threshold=0.0)

