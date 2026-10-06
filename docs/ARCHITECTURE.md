# Architecture

The project separates four concerns that are often tangled in fine-tuning
notebooks.

## Data plane

`RepairExample` is the supervised-training contract. It contains an instruction,
broken code, diagnostic context, and known-good target code. JSONL is used at
the boundary because it is streamable, diffable, and easy to validate.

`RepairTask` is the functional-evaluation contract. It contains broken code,
tests, and metadata, but never exposes the reference solution to generation.

## Training plane

The first trainer performs full-parameter supervised fine-tuning. It masks prompt
tokens with `-100`, calculates loss only on the target completion, accumulates
gradients, clips their norm, and applies AdamW. Checkpoints include model,
tokenizer, optimizer, scheduler, configuration, and trainer state.

## Evaluation plane

Generation and execution are deliberately separate. A model or external process
creates a JSONL prediction file. The evaluator matches predictions to tasks,
parses code, runs tests with a timeout, and emits structured results. This makes
it possible to reevaluate old generations without loading a model.

## Artifact plane

Every experiment receives its own output directory. Configuration files are
copied into it and reports refer to immutable task identifiers. Large artifacts
remain outside Git.
