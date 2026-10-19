"""Export the learned graph stage with explicit, spatially dynamic ONNX operators.

Group kernels and level-only FiLM are constants at inference. Keep six small
stage graphs so no shape-dependent Python branches enter the deployed runtime.
"""

import argparse
import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
from learned_centerline_model import load_weights
from learned_centerline_study import make_model
from onnx import TensorProto, helper, numpy_helper

from lineartgen.models.unet import progress_basis


class Builder:
    def __init__(self):
        self.nodes = []
        self.constants = []
        self.serial = 0

    def name(self, prefix):
        self.serial += 1
        return f"{prefix}_{self.serial}"

    def const(self, value):
        name = self.name("constant")
        self.constants.append(numpy_helper.from_array(np.asarray(value, np.float32), name))
        return name

    def op(self, kind, *inputs, **attrs):
        name = self.name(kind)
        self.nodes.append(helper.make_node(kind, list(inputs), [name], **attrs))
        return name

    def silu(self, x):
        return self.op("Mul", x, self.op("Sigmoid", x))

    def resize(self, x, factor, mode):
        if factor == 1:
            return x
        options = dict(mode=mode, coordinate_transformation_mode="half_pixel")
        if mode == "nearest":
            options.update(coordinate_transformation_mode="asymmetric", nearest_mode="floor")
        return self.op("Resize", x, "", self.const([1, 1, factor, factor]), **options)

    def conv(self, x, layer, level=None):
        dilation = layer.dilation if level is None else max(1, layer.dilation // 2**level)
        weight = np.asarray(layer.expanded_kernel()).transpose(3, 2, 0, 1)
        bias = np.tile(np.asarray(layer.bias[...]), 8)
        return self.op(
            "Conv",
            x,
            self.const(weight),
            self.const(bias),
            pads=[dilation] * 4,
            dilations=[dilation] * 2,
            kernel_shape=[3, 3],
        )

    def film(self, x, layer, level):
        basis = progress_basis(jnp.full((1, 1, 1, 1), level / 5, dtype=jnp.float32))
        gain, shift = np.split(np.asarray(layer.linear(basis)).ravel(), 2)
        gain = np.tile(1 + gain, 8).reshape(1, -1, 1, 1)
        shift = np.tile(shift, 8).reshape(1, -1, 1, 1)
        return self.op("Add", self.op("Mul", x, self.const(gain)), self.const(shift))


def build(model, level):
    b = Builder()
    guide, noise, insert = [
        b.op("Transpose", n, perm=[0, 3, 1, 2]) for n in ("guide", "noise_level", "insert")
    ]
    centered = b.op("Sub", guide, b.const(1.0))
    feedback = b.op("Sub", b.resize(insert, 2, "nearest"), b.const(1.0))
    features = b.op("Concat", centered, centered, noise, feedback, axis=1)
    stem = b.conv(features, model.stem, level)
    radius = max(1, 64 // 2**level)
    pooled = centered
    for kernel, pads in (
        ([2 * radius + 1, 1], [radius, 0, radius, 0]),
        ([1, 2 * radius + 1], [0, radius, 0, radius]),
    ):
        # ORT 1.24.4 miscounts padded AveragePool windows at image borders,
        # including when Pad + AveragePool is fused by its optimizer. Express
        # the training reduce-window sum and fixed divisor explicitly instead.
        pooled = b.op(
            "Conv",
            pooled,
            b.const(np.ones((1, 1, *kernel), np.float32)),
            kernel_shape=kernel,
            pads=pads,
            strides=[1, 1],
        )
        pooled = b.op("Div", pooled, b.const(2 * radius + 1))
    stem = b.op("Add", stem, b.conv(pooled, model.pooled_stem))
    factor = max(1, 16 // 2**level)
    coarse = (
        centered
        if factor == 1
        else b.op(
            "AveragePool",
            centered,
            kernel_shape=[factor, factor],
            strides=[factor, factor],
            count_include_pad=1,
        )
    )
    branch = model.coarse_branch
    for layer in (branch.lift, branch.first, branch.second):
        coarse = b.silu(b.conv(coarse, layer))
    weight = np.asarray(branch.project.kernel[...]).T[:, :, None, None]
    bias = np.asarray(branch.project.bias[...])
    coarse = b.op(
        "Conv",
        coarse,
        b.const(np.tile(weight, (8, 1, 1, 1))),
        b.const(np.tile(bias, 8)),
        kernel_shape=[1, 1],
        group=8,
    )
    stem = b.op("Add", stem, b.resize(coarse, factor, "linear"))
    x = b.silu(b.film(stem, model.stem_modulation, level))
    for block in model.blocks:
        y = b.silu(b.film(b.conv(x, block.first, level), block.modulation, level))
        x = b.silu(b.op("Add", x, b.conv(y, block.second, level)))
    block = model.context_block
    y = b.silu(b.film(b.conv(x, block.first, level), block.modulation, level))
    x = b.op("Add", x, b.conv(y, block.second, level))
    x = b.op(
        "Conv",
        x,
        b.const(np.asarray(model.head.kernel[...]).T[:, :, None, None]),
        b.const(np.asarray(model.head.bias[...])),
        kernel_shape=[1, 1],
    )
    b.nodes.append(helper.make_node("Transpose", [x], ["head"], perm=[0, 2, 3, 1]))
    inputs = [
        helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, "height", "width", 1])
        for n in ("guide", "noise_level")
    ]
    inputs.append(
        helper.make_tensor_value_info(
            "insert", TensorProto.FLOAT, [1, "half_height", "half_width", 1]
        )
    )
    output = helper.make_tensor_value_info("head", TensorProto.FLOAT, [1, "height", "width", 9])
    graph = helper.make_model(
        helper.make_graph(b.nodes, f"lineart_graph_level_{level}", inputs, [output], b.constants),
        opset_imports=[helper.make_opsetid("", 20)],
    )
    graph.ir_version = 10
    helper.set_model_props(
        graph,
        {
            "lineartgen": json.dumps(
                dict(
                    format_version=2, representation="attraction_graph", layout="NHWC", level=level
                )
            )
        },
    )
    onnx.checker.check_model(graph)
    return graph


def export(root, checkpoint, destination, total_steps):
    destination.mkdir(parents=True, exist_ok=True)
    model, _ = make_model(root)
    load_weights(model, checkpoint)
    comparisons, files = [], {}
    rng = np.random.default_rng(2026101911)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    for level in range(6):
        graph = build(model, level)
        p = destination / f"graph-level-{level}.onnx"
        onnx.save_model(graph, p)
        session = ort.InferenceSession(
            str(p), sess_options=options, providers=["CPUExecutionProvider"]
        )
        factor = max(2, 16 // 2**level)
        for height, width in ((factor * 2, factor * 3), (factor * 4, factor * 4)):
            guide = rng.uniform(0.2, 1, (1, height, width, 1)).astype(np.float32)
            noise = rng.uniform(0, 0.75, guide.shape).astype(np.float32)
            insert = rng.uniform(-1, 1, (1, height // 2, width // 2, 1)).astype(np.float32)
            expected = np.asarray(
                model(
                    jnp.asarray(guide),
                    jnp.asarray(noise),
                    jnp.asarray(insert),
                    jnp.asarray(guide),
                    jnp.ones_like(jnp.asarray(guide)),
                    pyramid_level=level,
                    return_network_output=True,
                )
            )
            actual = session.run(["head"], dict(guide=guide, noise_level=noise, insert=insert))[0]
            np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
            comparisons.append(
                dict(
                    level=level,
                    size=[height, width],
                    max_abs_error=float(np.max(np.abs(actual - expected))),
                )
            )
        files[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f"Validated level {level}", flush=True)
    metadata = dict(
        format_version=2,
        representation="attraction_graph",
        parameters=22725,
        levels=6,
        default_depth=3,
        halo=32,
        image_scale=1.0,
        radius=1.0,
        max_edge_length=2.0,
        supersampling=4,
        local_peak_window=33,
        training_steps=total_steps,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        source_checkpoint=str(checkpoint.relative_to(root)),
        files=files,
        validation=comparisons,
        description="Synthetic rough-to-line graph; experimental, residual double lines possible.",
    )
    (destination / "graph-model.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    args = parser.parse_args()
    export(
        args.root.resolve(), args.checkpoint.resolve(), args.destination.resolve(), args.total_steps
    )
