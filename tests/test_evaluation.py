from pathlib import Path

from posttrain_lab.data.schema import RepairTask
from posttrain_lab.eval.runner import evaluate_files, evaluate_task, summarize

ROOT = Path(__file__).parents[1]


def test_passing_repair() -> None:
    task = RepairTask(
        id="increment",
        instruction="Increment x.",
        buggy_code="def increment(x): return x",
        test_code=(
            "from solution import increment\n\n"
            "def test_increment(): assert increment(1) == 2\n"
        ),
        source="unit-test",
        split="test",
    )
    result = evaluate_task(task, "def increment(x):\n    return x + 1\n")
    assert result.parses
    assert result.passed


def test_syntax_error_is_diagnostic() -> None:
    task = RepairTask("bad", "Fix it", "pass", "def test_x(): pass", "unit-test", "test")
    result = evaluate_task(task, "def broken(:")
    assert not result.parses
    assert not result.tests_ran
    assert summarize([result])["pass_at_1"] == 0.0


def test_example_evaluation_end_to_end(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    payload = evaluate_files(
        ROOT / "data/examples/eval.jsonl",
        ROOT / "data/examples/predictions.jsonl",
        output,
    )
    assert payload["summary"]["pass_at_1"] == 1.0
    assert output.exists()
