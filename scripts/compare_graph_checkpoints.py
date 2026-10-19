"""Compare raw/EMA deployment candidates on existing development drawings only."""

import argparse
from pathlib import Path

import numpy as np
from evaluate_graph_preview import diagnostics, float_infer, panel, read, sha, write
from learned_centerline_study import PARTIAL
from lineartgen_runtime.graph import GraphLineartModel


def compare(candidates, inputs, output, limit):
    models = {}
    bindings = {}
    for candidate in candidates:
        name, directory = candidate.split("=", 1)
        path = Path(directory) / "graph-model.json"
        if name in models or not name.isidentifier():
            raise ValueError("Candidate labels must be distinct identifiers")
        models[name] = GraphLineartModel(path, provider="cpu")
        bindings[name] = dict(manifest_sha256=sha(path), **models[name].metadata)
    output.mkdir(parents=True, exist_ok=False)
    (output / "outputs").mkdir()
    (output / "native").mkdir()
    write(
        output / "binding.json",
        dict(candidates=bindings, inputs=str(inputs.resolve()), limit=limit),
    )
    records = read(inputs / "plan.json")["records"][:limit]
    rows, sources = [], {}
    for record in records:
        name = record["name"]
        path = inputs / record["file"]
        assert sha(path) == record["sha256"]
        with np.load(path) as z:
            source = z["source"]
        sources[name] = source
        arrays = {}
        for depth, mode in [(d, "whole") for d in (1, 3, 6)] + (
            [(3, "partial")] if name in PARTIAL else []
        ):
            for method, model in models.items():
                image = float_infer(model, source, depth, mode, 0.5)
                arrays[f"{method}-d{depth}-{mode}"] = image
                rows.append(
                    dict(name=name, method=method, depth=depth, mode=mode, **diagnostics(image))
                )
        np.savez_compressed(output / "outputs" / (name + ".npz"), **arrays)
        write(output / "progress.json", dict(cases=len(sources), predictions=len(rows)))
    pages = []
    for mode, depths in (("whole", (1, 3, 6)), ("partial", (3,))):
        for depth in depths:
            for size in (256, 512):
                names = [
                    n
                    for n, a in sources.items()
                    if a.shape[0] == size and (mode == "whole" or n in PARTIAL)
                ]
                count = 4 if size == 256 else 2
                for start in range(0, len(names), count):
                    page_rows = []
                    for name in names[start : start + count]:
                        with np.load(output / "outputs" / (name + ".npz")) as z:
                            arrays = [sources[name]] + [z[f"{m}-d{depth}-{mode}"] for m in models]
                        page_rows.append((name, arrays))
                    path = output / "native" / f"d{depth}-{mode}-{size}-{start // count:02d}.png"
                    panel(path, page_rows, ["source", *models])
                    pages.append(
                        dict(file=path.name, sha256=sha(path), cases=[p[0] for p in page_rows])
                    )
    write(
        output / "evaluation.json", dict(records=rows, used_for_selection=True, heldout_used=False)
    )
    write(output / "native/manifest.json", dict(records=pages, visually_reviewed=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True, dest="candidates")
    for name in ("inputs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--limit", type=int)
    compare(**vars(parser.parse_args()))
