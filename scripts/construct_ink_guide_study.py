"""Build/evaluate the predeclared within-encoding blend grid, never optimizer states."""

import argparse
import shutil
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from compare_corruption_mixture_study import paired_config, settings
from corruption_blend_study import run
from flax import nnx
from parameter_blend_study import freeze, original_source

from lineartgen.config import load_config, save_config
from lineartgen.models.blending import blend_parameters, compatible_config
from lineartgen.models.serialization import load_inference, save_inference


def prepare(root, coordinator, encoding):
    plan = read(coordinator / "plan.json")
    assert encoding in plan["encodings"] and plan["step"] == 2000
    assert plan["alpha_grid"] == [0, 1 / 16, 1 / 8, 1 / 4, 1]
    assert read(coordinator / "workflow-status.json")["stage"] == (
        "four_endpoints_complete_fixed_blends_pending"
    )
    assert not root.exists()
    endpoints, parents = {}, {}
    for family in ("strokes", "mixed"):
        declaration = plan["trained_endpoints"][family]
        parent, method = Path(declaration["root"]), declaration["methods"][encoding]
        original_source(parent)
        assert sha(parent / "plan.json") == sha(coordinator / "plan.json")
        assert read(parent / "checkpoint.json")["step"] == 2000
        audit = read(parent / "checkpoint-audits/step-2000.json")
        assert audit["snapshots_checked"] and audit["step"] == 2000
        record = next(r for r in audit["records"] if r["method"] == method)
        assert record["all_states_finite"] and record["parameters"] == 126828
        counterpart = plan["parents"][family]
        old = Path(counterpart["root"])
        original_source(old)
        assert sha(old / "source.json") == counterpart["source_sha256"]
        paired_config(
            settings(parent / f"configs/{method}.json", parent.parent.parent),
            settings(old / f"configs/{counterpart['method']}.json", old.parent.parent),
            {"model": {"guide_encoding": encoding}},
        )
        assert (
            read(parent / "checkpoint.json")["clean_sha256"]
            == read(old / "checkpoint.json")["clean_sha256"]
        )
        parents[family] = parent
        endpoints[family] = dict(
            root=str(parent), method=method, source_sha256=sha(parent / "source.json"), modes={}
        )
        for mode in ("ema", "raw"):
            folder = parent / f"snapshots/step-2000/{method}/{mode}"
            files = {n: sha(folder / n) for n in ("weights.npz", "metadata.json", "config.json")}
            assert record["snapshots"][mode]["checkpoint_exact"]
            assert files["weights.npz"] == record["snapshots"][mode]["weights_sha256"]
            endpoints[family]["modes"][mode] = dict(path=str(folder), files=files)
    assert len({read(p / "checkpoint.json")["clean_sha256"] for p in parents.values()}) == 1
    (root / "configs").mkdir(parents=True)
    base = parents["strokes"]
    for filename in (
        "dataset.json",
        "evaluation.json",
        "acceptance.json",
        "supplemental-panels.json",
        "confirmation-plan.json",
        "plan.json",
    ):
        shutil.copyfile(base / filename, root / filename)
    save_config(load_config(base / f"configs/{encoding}_strokes.json"), root / "config.json")
    write(
        root / "endpoint-seal.json",
        dict(
            endpoints=endpoints,
            step=2000,
            encoding=encoding,
            plan_sha256=sha(root / "plan.json"),
        ),
    )
    alphas = {
        f"{encoding}_strokes": 0,
        f"{encoding}_b0625": 1 / 16,
        f"{encoding}_b125": 1 / 8,
        f"{encoding}_b25": 1 / 4,
        f"{encoding}_mixed": 1,
    }
    assert list(alphas.values()) == plan["alpha_grid"]
    records = []
    for mode in ("ema", "raw"):
        first, a = load_inference(endpoints["strokes"]["modes"][mode]["path"])
        second, b = load_inference(endpoints["mixed"]["modes"][mode]["path"])
        compatible_config(a.model, b.model)
        assert a.model.guide_encoding == b.model.guide_encoding == encoding
        for name, alpha in alphas.items():
            model = blend_parameters(first, second, alpha)
            config = b if alpha == 1 else a
            folder = root / f"snapshots/step-2000/{name}/{mode}"
            save_inference(
                model,
                config,
                folder,
                provenance=(
                    f"Fixed {encoding} guidance study, alpha={alpha}, 2k, {mode}; "
                    "single parameter-blended network, no synthetic optimizer state"
                ),
            )
            loaded, saved = load_inference(folder)
            assert saved.model.guide_encoding == encoding
            jax.tree.map(np.testing.assert_array_equal, nnx.state(model), nnx.state(loaded))
            if alpha in (0, 1):
                jax.tree.map(
                    np.testing.assert_array_equal,
                    nnx.state(model),
                    nnx.state(first if alpha == 0 else second),
                )
            records.append(
                dict(
                    method=name,
                    mode=mode,
                    alpha=alpha,
                    encoding=encoding,
                    roundtrip_exact=True,
                    files={
                        n: sha(folder / n) for n in ("weights.npz", "metadata.json", "config.json")
                    },
                )
            )
            save_config(config, root / f"configs/{name}.json")
    control = Path(plan["control"]["root"])
    original_source(control)
    assert sha(control / "source.json") == plan["control"]["source_sha256"]
    baselines = [
        dict(root=str(control), methods=["control"], source_sha256=sha(control / "source.json"))
    ]
    for family in ("strokes", "mixed"):
        old = plan["parents"][family]
        baselines.append(
            dict(root=old["root"], methods=[old["method"]], source_sha256=old["source_sha256"])
        )
    original = read(base / "protocol.json")
    protocol = {
        k: original[k]
        for k in (
            "initial",
            "initial_files",
            "selected_panels",
            "total_schedule_steps",
            "patch_size",
        )
    }
    protocol.update(
        methods=list(alphas),
        milestones=[2000],
        step=2000,
        alpha_map=alphas,
        encoding=encoding,
        baselines=baselines,
        plan_sha256=sha(root / "plan.json"),
        endpoint_seal_sha256=sha(root / "endpoint-seal.json"),
        study_type="Strict final gates at fixed2k, combined selection across both encodings",
    )
    write(root / "protocol.json", protocol)
    write(
        root / "construction.json",
        dict(
            records=records,
            optimizer_states_created=False,
            script_sha256=sha(Path(__file__)),
            blend_source_sha256=sha(Path("src/lineartgen/models/blending.py")),
        ),
    )
    shutil.copytree(control / "evaluation/step-0-ema", root / "evaluation/step-0-ema")
    freeze(root)
    print(dict(root=str(root), encoding=encoding, models=list(alphas)), flush=True)


def rank(coordinator, roots):
    plan = read(coordinator / "plan.json")
    assert len(roots) == len(plan["encodings"])
    results, evidence, encodings = [], {}, []
    for root in roots:
        original_source(root)
        assert sha(root / "plan.json") == sha(coordinator / "plan.json")
        protocol = read(root / "protocol.json")
        encodings.append(protocol["encoding"])
        path = root / "quality-gate-with-baselines-2000.json"
        gate = read(path)
        assert gate["eligibility"] == "strict final despite 2k endpoint age"
        assert len(gate["results"]) == len(plan["alpha_grid"])
        assert all(not r["screening"] for r in gate["results"])
        for row in gate["results"]:
            results.append(
                dict(
                    row,
                    root=str(root),
                    encoding=protocol["encoding"],
                    alpha=protocol["alpha_map"][row["method"]],
                )
            )
        evidence[str(path)] = sha(path)
    assert sorted(encodings) == sorted(plan["encodings"])
    ranking = sorted(
        [r for r in results if r["numerical_gate_passed"]],
        key=lambda r: (-sum(r["primary_halo_gains"].values()), r["method"]),
    )
    write(
        coordinator / "combined-numerical-ranking.json",
        dict(
            results=results,
            ranked_candidates=ranking,
            evidence=evidence,
            plan_sha256=sha(coordinator / "plan.json"),
            visual_review_pending=True,
            selected_for_confirmation=None,
            production_adoption=False,
        ),
    )
    print([(r["method"], r["primary_halo_gains"]) for r in ranking], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "run", "rank"])
    p.add_argument("--root", type=Path)
    p.add_argument("--coordinator", type=Path)
    p.add_argument("--encoding", choices=["ink", "sqrt_ink"])
    p.add_argument("--roots", type=Path, nargs=2)
    args = p.parse_args()
    if args.action == "prepare":
        prepare(args.root.resolve(), args.coordinator.resolve(), args.encoding)
    elif args.action == "run":
        run(args.root.resolve())
    else:
        rank(args.coordinator.resolve(), [r.resolve() for r in args.roots])
