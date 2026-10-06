.PHONY: install test lint validate smoke

install:
	python -m pip install -e '.[dev]'

test:
	python -m pytest

lint:
	python -m ruff check .
	python -m mypy src

validate:
	posttrain validate-data data/examples/train.jsonl
	posttrain validate-tasks data/examples/eval.jsonl

smoke: test validate
