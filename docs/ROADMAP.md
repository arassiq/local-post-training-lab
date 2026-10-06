# Roadmap

## Phase 0 — Reproducible baseline

- Select and license-review a 0.5B–1.5B code-capable model.
- Benchmark untouched model on held-out Python repair tasks.
- Record latency, peak memory, parse rate, and pass@1.

## Phase 1 — Full-parameter SFT

- Train every parameter with AdamW on Apple Silicon.
- Add dataset fingerprints and complete checkpoint resumption.
- Compare validation loss with functional correctness.
- Publish the base-versus-SFT experiment report.

## Phase 2 — Data engine

- Add AST-aware mutation operators.
- Verify original-pass/mutant-fail behavior automatically.
- Split repositories before mutation generation.
- Add deduplication, provenance, and contamination checks.

## Phase 3 — Parameter-efficient baseline

- Add LoRA using the identical dataset and evaluator.
- Compare quality, memory, training time, and adapter size.
- Run rank, target-module, and dataset-size ablations.

## Phase 4 — Preference optimization

- Generate multiple candidates per task.
- Rank candidates using hidden tests and regression checks.
- Construct chosen/rejected pairs and run DPO.

## Phase 5 — Verifiable-reward RL

- Convert executable outcomes into a bounded reward.
- Penalize invalid code, regressions, and oversized changes.
- Monitor reward hacking and general-capability retention.

## Phase 6 — Autonomous coding system

- Add repository context selection and iterative repair.
- Route tasks across specialized adapters or checkpoints.
- Serve the system locally with tracing and resource budgets.
