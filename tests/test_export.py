import json

from flax import nnx
from lineartgen_runtime import LineartModel

from lineartgen.config import ExperimentConfig
from lineartgen.export import export_model
from lineartgen.models import UNet


def test_changed_architecture_exports_dynamic_shapes(tmp_path):
    config = ExperimentConfig.model_validate(
        {
            "model": {"channels": [3, 6, 12], "convolutions_per_block": 2},
        }
    )
    model = UNet(config.model, rngs=nnx.Rngs(12))
    output = export_model(model, config, tmp_path / "model.onnx")
    report = json.loads(output.with_suffix(".json").read_text())
    assert len(report["validation"]) == 3
    runtime = LineartModel(output, provider="cpu")
    width, height = 145, 129
    result = runtime.infer(
        scribble=bytes(width * height * 4),
        lineart=bytes(width * height * 4),
        width=width,
        height=height,
        strength=0.5,
        seed=42,
        denoise_steps=1,
    )
    assert len(result) == width * height * 4
