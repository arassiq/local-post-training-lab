from __future__ import annotations

import argparse
import json
from pathlib import Path

from posttrain_lab.data.schema import RepairExample, RepairTask, load_jsonl
from posttrain_lab.eval.runner import evaluate_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="posttrain")
    commands = parser.add_subparsers(dest="command", required=True)

    validate_data = commands.add_parser("validate-data", help="validate supervised JSONL")
    validate_data.add_argument("path", type=Path)

    validate_tasks = commands.add_parser("validate-tasks", help="validate evaluation JSONL")
    validate_tasks.add_argument("path", type=Path)

    train = commands.add_parser("train", help="run full-parameter AdamW SFT")
    train.add_argument("config", type=Path)

    generate = commands.add_parser("generate", help="generate one repair per task")
    generate.add_argument("--model", required=True)
    generate.add_argument("--revision", default="main")
    generate.add_argument("--tasks", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--max-new-tokens", type=int, default=256)
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--seed", type=int, default=17)

    evaluate = commands.add_parser("evaluate", help="execute and score predictions")
    evaluate.add_argument("--tasks", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--timeout", type=float, default=10.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "validate-data":
        records = load_jsonl(args.path, RepairExample)
        print(f"valid: {len(records)} supervised examples")
    elif args.command == "validate-tasks":
        records = load_jsonl(args.path, RepairTask)
        print(f"valid: {len(records)} evaluation tasks")
    elif args.command == "train":
        from posttrain_lab.train.sft import run_training

        run_training(args.config)
    elif args.command == "generate":
        from posttrain_lab.inference.generate import generate_predictions

        generate_predictions(
            args.model,
            args.tasks,
            args.output,
            revision=args.revision,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed,
        )
    elif args.command == "evaluate":
        payload = evaluate_files(args.tasks, args.predictions, args.output, args.timeout)
        print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
