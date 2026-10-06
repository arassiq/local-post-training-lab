"""
Test mutation operators and the mutation-data generation contract.

Topics to implement:
- Confirm every operator produces syntactically valid Python.
- Confirm an operator changes only its intended target.
- Confirm deterministic output under a fixed seed.
- Confirm unsupported source code produces no mutation rather than corrupt output.
- Confirm original programs pass and retained mutants fail their tests.
- Detect equivalent mutants and duplicate generated examples where possible.

Definitions:
- Property-based test: checks a general invariant across many generated inputs.
- Determinism: identical input and seed produce identical output.
- Invariant: a condition that must remain true across an operation.
"""
