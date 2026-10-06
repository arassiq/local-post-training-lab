# Evaluation contract

## Primary metric

`pass@1`: the fraction of tasks for which the first generated repair passes all
hidden tests.

## Diagnostic metrics

- Prediction coverage
- Non-empty output rate
- Python parse rate
- Test execution rate
- Full test pass rate
- Timeout rate
- Mean wall-clock execution time

Later versions will add pass@k, static analysis, regression tests, diff size,
general-capability retention, and bootstrap confidence intervals.

## Checkpoint selection

Validation cross-entropy diagnoses optimization and overfitting. Functional
validation selects the useful checkpoint. The held-out test set is evaluated
only after configuration and checkpoint selection are complete.

Tests do not directly affect the SFT gradient because execution is discrete and
non-differentiable. They can later produce chosen/rejected pairs for preference
optimization or scalar rewards for reinforcement learning.
