"""Krita's BGRA inference interface for the latent diffusion line-art preview."""

import hashlib
import json
import operator
import os
import threading
from pathlib import Path

import numpy as np

from .latent_edit import PixelEdit
from .latent_front import LatentFront
from .latent_painted_paths import edit_features


class LatentKritaModel:
    def __init__(self, manifest, num_threads=None, *, provider="auto", device_id=0):
        import onnxruntime as ort

        manifest = Path(manifest)
        release = json.loads(manifest.read_text(encoding="utf-8"))
        if release["format"] != "lineartgen_latent_edit_v1":
            raise ValueError("Unsupported latent edit model")
        folder = manifest.parent / "latent"
        for filename, digest in release["assets"].items():
            if Path(filename).name != filename:
                raise ValueError("Expected a model asset basename")
            if hashlib.sha256((folder / filename).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Latent model digest mismatch for {filename}")
        if provider not in ("auto", "directml", "cpu"):
            raise ValueError("provider must be auto, cpu or directml")
        available = "DmlExecutionProvider" in ort.get_available_providers()
        if provider == "directml" and not available:
            raise RuntimeError("DirectML is unavailable")
        providers = ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        threads = (
            min(4, os.cpu_count() or 1) if num_threads is None else operator.index(num_threads)
        )
        if threads < 1:
            raise ValueError("num_threads must be positive")
        options.intra_op_num_threads = threads
        if available and provider != "cpu":
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            providers.insert(0, ("DmlExecutionProvider", {"device_id": operator.index(device_id)}))
        self.front = LatentFront(folder, options, providers)
        self.merge = release["pixel_merge"]
        self.painted_paths = release.get("painted_paths", False)
        self.painted_loops = release.get("painted_loops", False)
        self.painted_tails = release.get("painted_tails", False)
        if self.merge:
            self.front.sessions["merge"] = ort.InferenceSession(
                str(folder / "merge.onnx"), sess_options=options, providers=providers
            )
        self.device = self.front.device
        self.backend = "onnxruntime latent diffusion (line-art preview)"
        self._lock = threading.Lock()

    def infer(self, *, scribble, lineart, width, height, strength, seed, denoise_steps):
        width, height = operator.index(width), operator.index(height)
        depth, seed = operator.index(denoise_steps), operator.index(seed)
        if min(width, height) < 1 or not 1 <= depth <= 6:
            raise ValueError("Expected positive image dimensions and one to six spatial levels")
        if not 0 <= seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        if not np.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("strength must be finite and between 0 and 1")
        edit = PixelEdit(scribble, lineart, height, width, strength)
        guide = edit.guide()
        if not edit.power.any() or not guide.any():
            return edit.render()
        with self._lock:
            proposal = self.front.generate(guide, edit.power, seed, depth)
            if self.merge and edit.alpha.any():
                inputs = edit_features(
                    edit,
                    proposal,
                    paths=self.painted_paths,
                    loops=self.painted_loops,
                    tails=self.painted_tails,
                )
                (under,) = self.front.run("merge", x=inputs)
                proposal = under[..., 0]
        return edit.render(proposal)
