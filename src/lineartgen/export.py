"""Export just the single-resolution forward, keeping the deployed runtime JAX-free."""

import json
from pathlib import Path

import jax
import numpy as np

from lineartgen.models import NestedUNet
from lineartgen.training import checkpoints


def export_model(model, config, destination, *, dynamic=True, provenance=None):
    import onnx
    import onnxruntime as ort
    from jax2onnx import to_onnx

    if isinstance(model, NestedUNet):
        if model.num_subnets != 1:
            raise ValueError(
                "Multi-prefix recursion needs its internal prediction channels. "
                "The current ONNX runtime supports only subnet 0; use extract for JAX inference."
            )
        model, model_config = model.to_unet()
        config = config.model_copy(update={"model": model_config})
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    minimum = config.model.minimum_size
    # Trace concrete shapes because jax2onnx's symbolic MaxPool/Resize adapters
    # are incomplete. The small, checked ONNX pass below makes this fully
    # convolutional graph spatially dynamic without changing the model code.
    size = minimum * 2
    image_shape, insert_shape = (1, size, size, 1), (1, size // 2, size // 2, 1)
    graph = to_onnx(
        model,
        [image_shape, image_shape, insert_shape],
        opset=20,
        model_name="lineartgen_unet",
        input_names=["image", "noise_level", "insert"],
        output_names=["prediction"],
    )
    # The graph uses ordinary tensors/operators; IR 10 also works with the ORT
    # releases supporting the plugin's Python 3.10 environment.
    graph.ir_version = 10
    if dynamic:
        make_spatially_dynamic(graph)
    metadata = {
        "format_version": 1,
        "layout": "NHWC",
        "minimum_size": minimum,
        "stop_size": max(128, minimum),
        "image_scale": config.data.image_scale,
        "model": config.model.model_dump(mode="json"),
        "dynamic": dynamic,
        "provenance": provenance or "JAX model",
    }
    onnx.helper.set_model_props(graph, {"lineartgen": json.dumps(metadata)})
    onnx.checker.check_model(graph)
    temporary = destination.with_suffix(".partial.onnx")
    onnx.save_model(graph, temporary)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(temporary), sess_options=options, providers=["CPUExecutionProvider"]
    )
    sizes = (
        [(minimum, minimum), (minimum * 2, minimum * 3), (minimum * 8, minimum * 8)]
        if dynamic
        else [(size, size)]
    )
    random = np.random.default_rng(42)
    comparisons = []
    for height, width in sizes:
        inputs = {
            "image": random.uniform(-1, 1, (1, height, width, 1)).astype(np.float32),
            "noise_level": random.uniform(0, 1, (1, height, width, 1)).astype(np.float32),
            "insert": random.uniform(-1, 1, (1, height // 2, width // 2, 1)).astype(np.float32),
        }
        expected = np.asarray(model(*(jax.numpy.asarray(x) for x in inputs.values())))
        actual = session.run(["prediction"], inputs)[0]
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
        comparisons.append(
            {"size": [height, width], "max_abs_error": float(np.abs(actual - expected).max())}
        )
    temporary.replace(destination)
    destination.with_suffix(".json").write_text(
        json.dumps({**metadata, "validation": comparisons}, indent=2) + "\n"
    )
    return destination


def make_spatially_dynamic(model):
    """Generalize this fully convolutional graph, with explicit safety checks.

    Conv/Relu/MaxPool/Concat/Transpose/Clip already accept dynamic spatial axes.
    Nearest Resize uses constant scale factors instead of a traced target size.
    Reject unfamiliar graph operations so a future architecture change cannot
    silently export a graph containing baked-in spatial dimensions.
    """
    import onnx
    from onnx import helper, numpy_helper

    supported = {
        "Conv",
        "Relu",
        "MaxPool",
        "Concat",
        "Transpose",
        "Resize",
        "Clip",
        "Identity",
        # Residual output uses only spatially pointwise arithmetic.
        "Sqrt",
        "Mul",
        "Add",
    }
    unsupported = {node.op_type for node in model.graph.node} - supported
    if unsupported:
        raise ValueError(
            f"Dynamic export pass needs review for new operators: {sorted(unsupported)}"
        )
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {
        value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        for value in [*inferred.graph.input, *inferred.graph.output, *inferred.graph.value_info]
    }
    for node in model.graph.node:
        if node.op_type != "Resize":
            continue
        attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if attributes.get("mode") != b"nearest":
            raise ValueError("Dynamic resize pass currently supports nearest-neighbor only")
        before, after = shapes[node.input[0]], shapes[node.output[0]]
        if len(before) != 4 or min(before) < 1 or len(after) != 4:
            raise ValueError("Resize requires known rank-four shapes during export")
        scales = np.asarray(after, np.float32) / np.asarray(before, np.float32)
        if not np.array_equal(scales, np.array([1, 2, 2, 1], np.float32)):
            raise ValueError(f"Expected NHWC 2x upsampling, got scales {scales}")
        name = node.output[0] + "_spatial_scales"
        model.graph.initializer.append(numpy_helper.from_array(scales, name))
        original = node.input[0]
        del node.input[:]
        node.input.extend([original, "", name])
    used = {name for node in model.graph.node for name in node.input}
    retained = [value for value in model.graph.initializer if value.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained)
    del model.graph.value_info[:]
    for value in [*model.graph.input, *model.graph.output]:
        dims = value.type.tensor_type.shape.dim
        if len(dims) != 4:
            raise ValueError("Expected rank-four image inputs and output")
        for axis, symbol in ((1, "height"), (2, "width")):
            dims[axis].ClearField("dim_value")
            dims[axis].dim_param = "insert_" + symbol if value.name == "insert" else symbol


def export_run(run_dir, destination, step=None, *, subnet=None):
    model, config, step = checkpoints.load_run_model(run_dir, step=step)
    if subnet is not None:
        if not isinstance(model, NestedUNet):
            raise ValueError("--subnet requires a nested model")
        model = model.extract(subnet)
    return export_model(model, config, destination, provenance=f"EMA at optimizer step {step}")
