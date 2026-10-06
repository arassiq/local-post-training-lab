from pathlib import Path

import pytest

from posttrain_lab.data.schema import RepairExample, RepairTask, load_jsonl

ROOT = Path(__file__).parents[1]


def test_example_files_are_valid() -> None:
    assert len(load_jsonl(ROOT / "data/examples/train.jsonl", RepairExample)) == 2
    assert len(load_jsonl(ROOT / "data/examples/eval.jsonl", RepairTask)) == 1


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    line = (ROOT / "data/examples/train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    path = tmp_path / "duplicate.jsonl"
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate id"):
        load_jsonl(path, RepairExample)


def test_prompt_masks_are_delimited() -> None:
    example = load_jsonl(ROOT / "data/examples/train.jsonl", RepairExample)[0]
    assert example.prompt().endswith("Corrected function:\n```python\n")
    assert example.completion().endswith("\n```")
