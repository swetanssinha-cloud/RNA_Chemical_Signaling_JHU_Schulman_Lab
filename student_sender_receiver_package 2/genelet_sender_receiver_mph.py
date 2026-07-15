#!/usr/bin/env python3
"""
Load-and-run helper for the sender/receiver COMSOL `.mph` model using MPh.

This script is aimed at the existing COMSOL model file, not at rebuilding the
entire model tree from scratch. That makes it much more reliable for handing to
a student who already has the COMSOL model.

Typical usage:

    python genelet_sender_receiver_mph.py \
        --mph-file /path/to/COMSOL-Model-SI-Chap-2-4.mph \
        --list-only

    python genelet_sender_receiver_mph.py \
        --mph-file /path/to/COMSOL-Model-SI-Chap-2-4.mph \
        --set x=300[um] \
        --set L=50[um] \
        --save-as reaction_diffusion_models/comsol_sender_receiver_run.mph

Notes:
- This script assumes a working COMSOL 6.4 installation and license.
- The model-specific parameter names must match what is already inside the
  `.mph` file. Use `--list-only` first to inspect available parameters/studies.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


COMSOL_BIN = Path("/Applications/COMSOL64/Multiphysics/bin/macarm64")
if COMSOL_BIN.is_dir():
    os.environ["PATH"] = f"{COMSOL_BIN}:{os.environ.get('PATH', '')}"

import mph  # noqa: E402


def parse_assignments(items: list[str]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f'Expected NAME=VALUE format, got "{item}".')
        name, value = item.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name or not value:
            raise ValueError(f'Expected NAME=VALUE format, got "{item}".')
        assignments[name] = value
    return assignments


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mph-file", type=Path, required=True, help="Existing COMSOL model file.")
    parser.add_argument("--save-as", type=Path, default=None, help="Path for the solved output model.")
    parser.add_argument("--study", type=str, default=None, help="Optional study name to solve.")
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset name for evaluation.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Override a global COMSOL parameter.",
    )
    parser.add_argument(
        "--eval",
        action="append",
        default=[],
        metavar="EXPR",
        help="Evaluate an expression after solving. Repeat for multiple expressions.",
    )
    parser.add_argument(
        "--inner",
        type=str,
        default="last",
        help='Inner solution index for evaluation: "last", "first", or an integer string.',
    )
    parser.add_argument("--cores", type=int, default=1, help="COMSOL cores to request.")
    parser.add_argument("--list-only", action="store_true", help="Print model contents and exit.")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional JSON summary output for parameters/studies/datasets/evaluations.",
    )
    parser.add_argument(
        "--export-existing",
        action="store_true",
        help="Run all pre-existing export nodes after solve.",
    )
    return parser.parse_args()


def maybe_parse_inner(value: str):
    if value in ("first", "last"):
        return value
    return int(value)


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(input_path.stem + "_codex_run" + input_path.suffix)


def main():
    args = parse_args()
    if not args.mph_file.exists():
        raise FileNotFoundError(f"Model file not found: {args.mph_file}")

    overrides = parse_assignments(args.set)
    inner = maybe_parse_inner(args.inner)

    client = mph.start(cores=args.cores)
    model = None
    summary: dict[str, object] = {}
    try:
        model = client.load(args.mph_file)

        parameters = model.parameters()
        studies = model.studies()
        datasets = model.datasets()

        print(f"Loaded model: {args.mph_file}")
        print(f"Studies: {studies}")
        print(f"Datasets: {datasets}")
        print(f"Parameters ({len(parameters)} total):")
        for name in sorted(parameters):
            print(f"  {name} = {parameters[name]}")

        summary["model_file"] = str(args.mph_file)
        summary["studies"] = studies
        summary["datasets"] = datasets
        summary["parameters"] = parameters

        if args.list_only:
            if args.summary_json:
                args.summary_json.parent.mkdir(parents=True, exist_ok=True)
                args.summary_json.write_text(json.dumps(summary, indent=2))
                print(f"Saved summary JSON to {args.summary_json}")
            return

        if overrides:
            print("Applying parameter overrides:")
            for name, value in overrides.items():
                print(f"  {name} = {value}")
                model.parameter(name, value)
            summary["overrides"] = overrides

        print("Solving model...")
        if args.study:
            model.solve(args.study)
        else:
            model.solve()
        print("Solve complete.")

        evaluations: dict[str, object] = {}
        if args.eval:
            for expr in args.eval:
                value = model.evaluate(expr, dataset=args.dataset, inner=inner)
                evaluations[expr] = value.tolist() if hasattr(value, "tolist") else value
                print(f"{expr} -> {evaluations[expr]}")
        if evaluations:
            summary["evaluations"] = evaluations

        if args.export_existing:
            print("Running existing COMSOL export nodes...")
            model.export()
            print("Finished running export nodes.")

        output_path = args.save_as or default_output_path(args.mph_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(output_path)
        print(f"Saved solved model to {output_path}")
        summary["saved_model"] = str(output_path)

        if args.summary_json:
            args.summary_json.parent.mkdir(parents=True, exist_ok=True)
            args.summary_json.write_text(json.dumps(summary, indent=2))
            print(f"Saved summary JSON to {args.summary_json}")
    finally:
        if model is not None:
            try:
                client.remove(model)
            except Exception:
                pass
        try:
            client.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
