from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

TrainingSplit = Literal["train", "validation"]
EvaluationSplit = Literal["validation", "test"]


@dataclass(frozen=True)
class RepairExample:
    id: str
    instruction: str
    buggy_code: str
    test_output: str
    target_code: str
    source: str
    split: TrainingSplit

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RepairExample:
        required = {field.name for field in cls.__dataclass_fields__.values()}
        _validate_keys(value, required, "repair example")
        example = cls(**value)
        _validate_non_empty(asdict(example), required - {"test_output"}, example.id)
        if example.split not in {"train", "validation"}:
            raise ValueError(f"{example.id}: invalid training split {example.split!r}")
        return example

    def prompt(self) -> str:
        return (
            "Repair the Python function. Return only the complete corrected function.\n\n"
            f"Instruction:\n{self.instruction}\n\n"
            f"Broken code:\n```python\n{self.buggy_code.rstrip()}\n```\n\n"
            f"Test output:\n{self.test_output.strip() or '(not provided)'}\n\n"
            "Corrected function:\n```python\n"
        )

    def completion(self) -> str:
        return f"{self.target_code.rstrip()}\n```"


@dataclass(frozen=True)
class RepairTask:
    id: str
    instruction: str
    buggy_code: str
    test_code: str
    source: str
    split: EvaluationSplit

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RepairTask:
        required = {field.name for field in cls.__dataclass_fields__.values()}
        _validate_keys(value, required, "repair task")
        task = cls(**value)
        _validate_non_empty(asdict(task), required, task.id)
        if task.split not in {"validation", "test"}:
            raise ValueError(f"{task.id}: invalid evaluation split {task.split!r}")
        return task

    def prompt(self) -> str:
        return (
            "Repair the Python function. Return only the complete corrected function.\n\n"
            f"Instruction:\n{self.instruction}\n\n"
            f"Broken code:\n```python\n{self.buggy_code.rstrip()}\n```\n\n"
            "Corrected function:\n```python\n"
        )


RecordT = TypeVar("RecordT", RepairExample, RepairTask)


def load_jsonl(path: Path, record_type: type[RecordT]) -> list[RecordT]:
    records: list[RecordT] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            record = record_type.from_dict(value)
            if record.id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate id {record.id!r}")
            seen.add(record.id)
            records.append(record)
    if not records:
        raise ValueError(f"{path}: dataset is empty")
    return records


def _validate_keys(value: dict[str, Any], required: set[str], label: str) -> None:
    missing = required - value.keys()
    extra = value.keys() - required
    if missing:
        raise ValueError(f"{label} missing fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"{label} has unknown fields: {sorted(extra)}")


def _validate_non_empty(value: dict[str, Any], fields: set[str], record_id: str) -> None:
    empty = [name for name in fields if not isinstance(value[name], str) or not value[name].strip()]
    if empty:
        raise ValueError(f"{record_id or '<unknown>'}: empty fields: {sorted(empty)}")
