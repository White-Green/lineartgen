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
    if study != "methods" or baseline:
        raise ValueError("Use study=widths or methods; --baseline is only for widths")
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
