"""Reusable ONNX session implementing the existing LineartModel.infer API."""

import json
import operator
import threading
from pathlib import Path

import numpy as np

from .images import prepare_input, pyramid_plan, to_bgra
from .sampling import denoise


class LineartModel:
    def __init__(self, model_path=None, num_threads=None, *, provider="auto", device_id=0):
        graph_path = (
            Path(model_path)
            if model_path is not None
            else Path(__file__).parent / "assets" / "graph-model.json"
        )
        self._graph = None
        if graph_path.name == "graph-model.json" and graph_path.is_file():
            from .graph import GraphLineartModel

            self._graph = GraphLineartModel(
                graph_path, num_threads, provider=provider, device_id=device_id
            )
            self.backend, self.device = self._graph.backend, self._graph.device
            return
        import onnxruntime as ort

        path = (
            Path(model_path)
            if model_path is not None
            else Path(__file__).parent / "assets" / "model.onnx"
        )
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {path}")
        available = ort.get_available_providers()
        if provider not in ("auto", "cpu", "directml"):
            raise ValueError("provider must be auto, cpu or directml")
        use_dml = provider != "cpu" and "DmlExecutionProvider" in available
        if provider == "directml" and not use_dml:
            raise RuntimeError(
                "DirectML is unavailable; install a Windows package with onnxruntime-directml"
            )
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1 if num_threads is None else operator.index(num_threads)
        if options.intra_op_num_threads < 1:
            raise ValueError("num_threads must be positive")
        providers = ["CPUExecutionProvider"]
        if use_dml:
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            providers.insert(0, ("DmlExecutionProvider", {"device_id": device_id}))
        self._session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        metadata = self._session.get_modelmeta().custom_metadata_map
        try:
            self._metadata = json.loads(metadata["lineartgen"])
        except (KeyError, ValueError) as error:
            raise ValueError("Model is missing lineartgen export metadata") from error
        version = self._metadata.get("format_version")
        if version not in (1, 2) or self._metadata.get("layout") != "NHWC":
            raise ValueError("Unsupported lineartgen model format")
        inputs = {value.name for value in self._session.get_inputs()}
        self.source_conditioning = self._metadata.get("source_conditioning", False)
        self.progress_conditioning = self._metadata.get("progress_conditioning", False)
        if self.progress_conditioning and not self.source_conditioning:
            raise ValueError("Progress conditioning requires source conditioning")
        self._preserve_zero_noise = (
            self._metadata.get("model", {}).get("output_mode", "direct") == "residual"
        )
        model_config = self._metadata.get("model", {})
        self.sampling_mode = model_config.get("sampling_mode", "diffusion")
        if self.sampling_mode not in ("diffusion", "one_pass"):
            raise ValueError("Unsupported sampling mode")
        if (version == 2) != (self.sampling_mode == "one_pass"):
            raise ValueError("One-pass restoration requires format version 2")
        if self.sampling_mode == "one_pass" and (
            not self.source_conditioning
            or (self.progress_conditioning and model_config.get("architecture") != "local")
            or not self._preserve_zero_noise
            or model_config.get("feature_input") != "source"
            or model_config.get("residual_base") != "source"
        ):
            raise ValueError("One-pass restoration requires original-source features and anchor")
        expected_inputs = {"image", "noise_level", "insert"}
        if self.source_conditioning:
            expected_inputs.add("guide")
        if self.progress_conditioning:
            expected_inputs.add("progress")
        if inputs != expected_inputs:
            raise ValueError(f"Unexpected ONNX input contract: {sorted(inputs)}")
        self.backend = "onnxruntime"
        self.device = ", ".join(self._session.get_providers())
        self._lock = threading.Lock()

    def forward(self, image, noise_level, insert, guide=None, progress=None):
        if self._graph is not None:
            raise ValueError("Graph models use per-level heads; call infer for BGRA images")
        inputs = {"image": image, "noise_level": noise_level, "insert": insert}
        if self.source_conditioning:
            if guide is None:
                raise ValueError("This model requires the original rough guide")
            inputs["guide"] = guide
        elif guide is not None:
            raise ValueError("This model does not accept a guide")
        if self.progress_conditioning:
            if progress is None:
                raise ValueError("This model requires remaining progress")
            inputs["progress"] = progress
        elif progress is not None:
            raise ValueError("This model does not accept progress")
        return self._session.run(["prediction"], inputs)[0]

    def infer(self, *, scribble, lineart, width, height, strength, seed, denoise_steps):
        """Restore BGRA input; one_pass models use one pass for any valid step request."""
        if self._graph is not None:
            return self._graph.infer(
                scribble=scribble,
                lineart=lineart,
                width=width,
                height=height,
                strength=strength,
                seed=seed,
                denoise_steps=denoise_steps,
            )
        width, height = operator.index(width), operator.index(height)
        seed, denoise_steps = operator.index(seed), operator.index(denoise_steps)
        if not np.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("strength must be finite and between 0 and 1")
        if not 1 <= denoise_steps <= 20:
            raise ValueError("denoise_steps must be between 1 and 20")
        if not 0 <= seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        plan = pyramid_plan(
            height, width, self._metadata["minimum_size"], self._metadata["stop_size"]
        )
        clean, noise = prepare_input(
            scribble, lineart, plan, strength, self._metadata["image_scale"]
        )
        with self._lock:
            prediction = denoise(
                self.forward,
                clean,
                noise,
                seed=seed,
                steps=denoise_steps,
                levels=len(plan.sizes),
                source_conditioning=self.source_conditioning,
                progress_conditioning=self.progress_conditioning,
                preserve_zero_noise=self._preserve_zero_noise,
                sampling_mode=self.sampling_mode,
            )
        if not np.isfinite(prediction).all():
            raise RuntimeError("ONNX inference produced nonfinite output")
        return to_bgra(prediction, plan)
