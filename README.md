# Local Post-Training Lab

A local-first laboratory for **full-parameter fine-tuning and execution-based
evaluation of code models on Apple Silicon**.

The first research question is deliberately narrow:

> Can full supervised fine-tuning with AdamW improve a roughly 1B-parameter
> model's ability to repair broken Python functions without degrading its
> broader coding behavior?

This repository is scaffolded as an experiment, not a chatbot demo. It keeps
training data, optimization, functional evaluation, and experiment artifacts
separate so that every result can be reproduced and challenged.

## First milestone

The initial comparison has two arms:

| Arm | Parameters updated | Objective |
| --- | --- | --- |
| Base | None | Establish the pre-training baseline |
| Full SFT | All model parameters | Token-level cross-entropy optimized with AdamW |

Later milestones add LoRA, preference optimization, and reinforcement learning
with executable test rewards. They are intentionally not part of the first
experiment.

## System shape

```text
Verified repair examples
        │
        ├── train ──> tokenize + mask prompt ──> AdamW SFT ──> checkpoints
        │
        └── eval ──> generate candidate code ──> isolated test runner
                                                    │
                                                    └──> metrics + report
```

During SFT, tests do **not** participate directly in backpropagation. The
training objective is next-token cross-entropy on target code. Executable tests
are used to validate data, compare checkpoints, and select the final model.
Later, those same signals can create preference pairs or reinforcement-learning
rewards.

## Repository layout

```text
configs/                 Reproducible experiment configurations
data/examples/           Tiny committed fixtures for smoke tests
docs/                    Architecture, data, evaluation, and roadmap notes
src/posttrain_lab/data/  Schemas, validation, and mutation generation
src/posttrain_lab/train/ Full-parameter AdamW SFT implementation
src/posttrain_lab/eval/  Execution runner and aggregate metrics
tests/                   Unit and end-to-end smoke tests
```

## Installation

Use Python 3.11 or 3.12. A virtual environment is strongly recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

PyTorch's MPS backend is selected automatically on a compatible Mac. Confirm it
before a long run:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Validate the example data

```bash
posttrain validate-data data/examples/train.jsonl
posttrain validate-tasks data/examples/eval.jsonl
```

## Run a training experiment

Edit `configs/sft_1b.yaml` and set `model.name` to the licensed model checkpoint
you have selected. Then run:

```bash
posttrain train configs/sft_1b.yaml
```

The default configuration is conservative: 1,024-token sequences, a micro-batch
of one, gradient accumulation, BF16 when available, and gradient checkpointing.
Start with `max_steps: 20` as a plumbing test before committing to a full run.

## Evaluate generated repairs

Model generation and code execution are separated. Generate base-model or
checkpoint predictions with:

```bash
posttrain generate \
  --model MODEL_OR_CHECKPOINT \
  --tasks data/examples/eval.jsonl \
  --output artifacts/predictions.jsonl
```

Prediction files use JSONL:

```json
{"task_id":"add-boundary","generated_code":"def clamp(value, low, high):\n    return max(low, min(value, high))\n"}
```

Then evaluate them:

```bash
posttrain evaluate \
  --tasks data/examples/eval.jsonl \
  --predictions artifacts/predictions.jsonl \
  --output reports/example.json
```

> [!CAUTION]
> The local evaluator executes generated Python. Its temporary directory and
> timeout provide operational isolation, **not a security boundary**. Only run
> trusted code locally. A containerized executor is planned before evaluating
> arbitrary third-party generations.

## Experimental rules

1. Split by repository or problem family before generating mutations.
2. Never tune against the held-out test set.
3. Record the exact model revision, tokenizer, seed, configuration, and dataset fingerprint.
4. Compare functional outcomes, not only validation loss.
5. Preserve failed experiments and explain what they taught us.

See [docs/ROADMAP.md](docs/ROADMAP.md) for the staged plan and
[docs/EVALUATION.md](docs/EVALUATION.md) for the metric contract.

## Status

The repository currently provides the experiment scaffold, a custom
full-parameter SFT loop, dataset validation, deterministic mutation primitives,
an executable repair evaluator, and tests. The next step is selecting the first
model checkpoint and producing the initial base-model benchmark.

## License

MIT. Model weights and datasets retain their own licenses and must be reviewed
before use or redistribution.
