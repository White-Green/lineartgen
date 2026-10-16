"""Extend only half of already augmented wide-source components to32--64px."""

import numpy as np
from continuous_width_data import width_choice, width_source
from contrast_composition_data import overlay_source


def extended_width(seed, step, item, index, original, *, component=0):
    if index < 464 or original is None:
        return original
    rng = np.random.default_rng([seed, step, item, 0x57584558, component])
    if rng.random() >= 0.5:
        return original
    return float(np.exp(rng.uniform(np.log(32), np.log(64))))


def width_records(selection, seed, step):
    rows = []
    for item, (index, overlay) in enumerate(
        zip(selection["indices"], selection["overlays"], strict=True)
    ):
        old = width_choice(seed, step, item, index)
        new = extended_width(seed, step, item, index, old)
        row = dict(item=item, primary=dict(index=index, old=old, new=new))
        if overlay["enabled"]:
            index, old = overlay["index"], overlay["width"]
            new = extended_width(seed, step, item, index, old, component=1)
            row["overlay"] = dict(index=index, old=old, new=new)
        rows.append(row)
    return rows


def sources(data, selection, records, *, expanded):
    result, clean = [], []
    for item, (index, opts, overlay) in enumerate(
        zip(selection["indices"], selection["augmentation"], selection["overlays"], strict=True)
    ):
        row = records[item]
        key = "new" if expanded else "old"
        source = width_source(data, data["records"][index], index, row["primary"][key], opts)
        target = data["variants"][index, 13]
        if overlay["enabled"]:
            j = overlay["index"]
            other = width_source(
                data, data["records"][j], j, row["overlay"][key], overlay["augmentation"]
            )
            source = overlay_source(source, other)
            target = np.minimum(target, data["variants"][j, 13])
        result.append(source)
        clean.append(target)
    return np.stack(result), np.stack(clean)


def changed_examples(records):
    return [any(r["old"] != r["new"] for k, r in row.items() if k != "item") for row in records]
