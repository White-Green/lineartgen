"""Small CLI; subcommands import only the components they need."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="lineartgen")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train or resume an experiment")
    train.add_argument("--config", type=Path)
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--resume", action="store_true")
    train.add_argument(
        "--stop-after", type=int, help="Stop after this many additional optimizer steps"
    )
    export = commands.add_parser("export", help="Export EMA weights and validate ONNX against JAX")
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--step", type=int)
    export.add_argument("--subnet", type=int, help="Nested subnet (currently 0 for ONNX)")
    extract = commands.add_parser(
        "extract", help="Save a standalone nested prefix for JAX inference"
    )
    extract.add_argument("--run-dir", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--subnet", type=int, required=True)
    extract.add_argument("--step", type=int)
    prepare = commands.add_parser(
        "prepare", help="Create an optional lossless tiled TIFF crop cache"
    )
    prepare.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--limit", type=int, help="Only prepare this many source images")
    commands.add_parser("doctor", help="Print versions and visible training devices")
    args = parser.parse_args(argv)
    if args.command == "train":
        from lineartgen.config import load_config
        from lineartgen.training.trainer import Trainer

        if args.stop_after is not None and args.stop_after < 1:
            parser.error("--stop-after must be positive")
        config_path = args.config or (
            args.run_dir / "config.json" if args.resume else Path("configs/default.toml")
        )
        Trainer(load_config(config_path), args.run_dir).fit(
            resume=args.resume, stop_after=args.stop_after
        )
    elif args.command == "export":
        from lineartgen.export import export_run

        print(export_run(args.run_dir, args.output, args.step, subnet=args.subnet))
    elif args.command == "extract":
        from lineartgen.models.serialization import extract_run

        print(extract_run(args.run_dir, args.output, args.subnet, step=args.step))
    elif args.command == "prepare":
        from lineartgen.config import load_config
        from lineartgen.data.pipeline import prepare_dataset

        for output in prepare_dataset(load_config(args.config).data, args.output, args.limit):
            print(output, flush=True)
    elif args.command == "doctor":
        import importlib.metadata

        import jax

        for package in ("jax", "jaxlib", "flax", "optax", "orbax-checkpoint", "grain"):
            print(f"{package}: {importlib.metadata.version(package)}")
        print(f"Devices: {jax.devices()}")


if __name__ == "__main__":
    main()
