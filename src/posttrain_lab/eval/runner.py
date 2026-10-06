from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from posttrain_lab.data.schema import RepairTask, load_jsonl


@dataclass(frozen=True)
class EvaluationResult:
    task_id: str
    has_prediction: bool
    parses: bool
    tests_ran: bool
    passed: bool
    timed_out: bool
    duration_seconds: float
    output: str


def evaluate_files(
    task_path: Path,
    prediction_path: Path,
    output_path: Path,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    tasks = load_jsonl(task_path, RepairTask)
    predictions = _load_predictions(prediction_path)
    results = [evaluate_task(task, predictions.get(task.id), timeout_seconds) for task in tasks]
    summary = summarize(results)
    payload = {"summary": summary, "results": [asdict(result) for result in results]}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def evaluate_task(
    task: RepairTask, generated_code: str | None, timeout_seconds: float = 10.0
) -> EvaluationResult:
    if generated_code is None or not generated_code.strip():
        return EvaluationResult(task.id, False, False, False, False, False, 0.0, "")
    try:
        ast.parse(generated_code)
    except SyntaxError as exc:
        return EvaluationResult(task.id, True, False, False, False, False, 0.0, str(exc))

    with tempfile.TemporaryDirectory(prefix="posttrain-eval-") as directory:
        root = Path(directory)
        (root / "solution.py").write_text(generated_code, encoding="utf-8")
        (root / "test_solution.py").write_text(task.test_code, encoding="utf-8")
        started = time.monotonic()
        try:
            process = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "test_solution.py"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - started
            output = f"{exc.stdout or ''}\n{exc.stderr or ''}".strip()
            return EvaluationResult(task.id, True, True, True, False, True, duration, output)
        duration = time.monotonic() - started
        output = f"{process.stdout}\n{process.stderr}".strip()
        return EvaluationResult(
            task.id, True, True, True, process.returncode == 0, False, duration, output
        )


def summarize(results: list[EvaluationResult]) -> dict[str, float | int]:
    total = len(results)
    return {
        "tasks": total,
        "prediction_coverage": _rate(results, "has_prediction"),
        "parse_rate": _rate(results, "parses"),
        "execution_rate": _rate(results, "tests_ran"),
        "pass_at_1": _rate(results, "passed"),
        "timeout_rate": _rate(results, "timed_out"),
        "mean_duration_seconds": (
            sum(result.duration_seconds for result in results) / total if total else 0.0
        ),
    }


def _rate(results: list[EvaluationResult], field: str) -> float:
    if not results:
        return 0.0
    return sum(bool(getattr(result, field)) for result in results) / len(results)


def _load_predictions(path: Path) -> dict[str, str]:
    predictions: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if set(value) != {"task_id", "generated_code"}:
                raise ValueError(f"{path}:{line_number}: invalid prediction fields")
            task_id = value["task_id"]
            if task_id in predictions:
                raise ValueError(f"{path}:{line_number}: duplicate task_id {task_id!r}")
            predictions[task_id] = value["generated_code"]
    return predictions
