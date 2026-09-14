"""Named, single-factor comparisons with a fixed optimizer batch and random pool."""

from lineartgen.config import ModelConfig


def comparison_configs(config, *, study="widths", baseline=False):
    if config.model.architecture != "nested" or not config.model.nested_channels:
        raise ValueError("A nested model with at least two widths is required")
    if study == "widths":
        result = {
            "shared": config,
            "independent_core": config.model_copy(
                update={"model": config.model.model_copy(update={"nested_channels": ()})}
            ),
        }
        if baseline:
            result["baseline"] = config.model_copy(update={"model": ModelConfig()})
        return result
    if study == "boundary_loss":
        if (
            baseline
            or config.model.output_mode != "residual"
            or config.matching.method != "independent"
            or config.sketch.mode != "strokes"
            or config.geometry.probability != 0.2
            or config.loss.boundary_weight <= 0
            or config.diffusion.pattern_weights[3]
        ):
            raise ValueError(
                "Boundary comparison requires geometry20, strokes, residual, "
                "independent noise/masks and a calibrated positive boundary weight"
            )
        return {
            "control": config.model_copy(
                update={"loss": config.loss.model_copy(update={"boundary_weight": 0.0})}
            ),
            "boundary": config,
        }
    if study == "geometry_data":
        if (
            baseline
            or config.model.output_mode != "residual"
            or config.matching.method != "independent"
            or config.sketch.mode != "strokes"
            or config.geometry.probability != 0
            or config.diffusion.pattern_weights[3]
        ):
            raise ValueError(
                "Geometry data control requires residual, independent noise, "
                "strokes and image-independent masks, with geometry disabled"
            )
        return {
            "control": config,
            "geometry20": config.model_copy(
                update={"geometry": config.geometry.model_copy(update={"probability": 0.2})}
            ),
        }
    if study in ("sketch", "sketch_rates"):
        if baseline or config.model.output_mode != "residual":
            raise ValueError("The sketch study requires residual output and no width baseline")
        if config.matching.method != "independent" or config.sketch.mode != "none":
            raise ValueError("The sketch control must use independent noise and no degradation")
        if config.diffusion.pattern_weights[3]:
            raise ValueError("Use image-independent mask patterns for paired sketch comparisons")
        if study == "sketch_rates":
            return {
                name: config.model_copy(
                    update={
                        "sketch": config.sketch.model_copy(
                            update={"mode": "strokes", "probability": probability}
                        )
                    }
                )
                for name, probability in (("p75", 0.75), ("p50", 0.50), ("p25", 0.25))
            }
        return {
            name: config.model_copy(
                update={"sketch": config.sketch.model_copy(update={"mode": mode})}
            )
            for name, mode in (
                ("control", "none"),
                ("tone", "tone"),
                ("blur", "blur"),
                ("strokes", "strokes"),
            )
        }
    if study != "methods" or baseline:
        raise ValueError(
            "Use study=widths, methods, sketch or sketch_rates; --baseline is only for widths"
        )
    if config.model.output_mode != "direct" or config.matching.method != "exact":
        raise ValueError("The methods control must use direct output and exact OT")
    batch = config.train.effective_batch_size
    generated = config.matching.generation_pool_size or config.matching.pool_size
    result = {
        "current": config,
        "residual": config.model_copy(
            update={"model": config.model.model_copy(update={"output_mode": "residual"})}
        ),
    }
    for name, method in (("ot_full", "exact"), ("independent_noise", "independent")):
        result[name] = config.model_copy(
            update={
                "matching": config.matching.model_copy(
                    update={"method": method, "pool_size": batch, "generation_pool_size": generated}
                )
            }
        )
    return result
