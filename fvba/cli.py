"""Command-line interface for FVBA experiments."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from fvba.config import load_config
from fvba.data.cifar import SyntheticClassificationDataset, build_cifar10
from fvba.data.imagefolder import build_imagefolder
from fvba.data.imagefolder import validate_imagefolder


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FVBA image-classification experiments")
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser("prepare-data", help="download or verify dataset")
    prepare.add_argument("--config", type=Path, required=True)
    run = subcommands.add_parser("run", help="run or resume an experiment")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--seed", type=int)
    run.add_argument("--victim", choices=("resnet18", "vgg19"))
    run.add_argument("--output-dir", type=Path)
    evaluate = subcommands.add_parser("evaluate", help="inspect completed metrics")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    return parser


def _prepare_data(config: dict) -> None:
    data = config["data"]
    if data["dataset"] == "synthetic":
        SyntheticClassificationDataset(
            size=int(data.get("synthetic_size", 32)),
            num_classes=int(data["num_classes"]),
            seed=int(config["runtime"]["seed"]),
        )
        return
    if data["dataset"] in {"animals90", "imagenet100"}:
        counts = validate_imagefolder(data["root"], num_classes=int(data["num_classes"]))
        build_imagefolder(data["root"], train=True, augment=False, image_size=int(data["image_size"]))
        build_imagefolder(data["root"], train=False, augment=False, image_size=int(data["image_size"]))
        print(f"Data ready: train={counts['train']} test={counts['test']}")
        return
    if data["dataset"] != "cifar10":
        raise ValueError(f"Unsupported dataset: {data['dataset']}")
    try:
        build_cifar10(
            data["root"], train=True, augment=False, download=bool(data["download"])
        )
        build_cifar10(
            data["root"], train=False, augment=False, download=bool(data["download"])
        )
    except RuntimeError as error:
        raise RuntimeError(
            f"CIFAR-10 is unavailable at {data['root']}; set data.download=true "
            "or place the extracted dataset there"
        ) from error


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare-data":
        config = load_config(args.config)
        _prepare_data(config)
        print("Data ready")
        return 0
    if args.command == "run":
        config = deepcopy(load_config(args.config))
        if args.seed is not None:
            config["runtime"]["seed"] = args.seed
        if args.victim is not None:
            config.setdefault("model", {})["victim"] = args.victim
        if args.output_dir is not None:
            config.setdefault("output", {})["dir"] = str(args.output_dir)
        if args.dry_run:
            print("Configuration valid")
            return 0
        from fvba.pipeline import build_pipeline_phases
        from fvba.orchestrator import RunOrchestrator

        output = RunOrchestrator(
            config, phases=build_pipeline_phases(config)
        ).run()
        print(output)
        return 0
    run_dir = args.run_dir
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Run directory has no metrics.json: {run_dir}")
    print(json.dumps(json.loads(metrics_path.read_text(encoding="utf-8")), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
