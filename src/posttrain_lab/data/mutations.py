"""
Generate controlled bugs from correct Python programs for repair-model training data.

Topics to implement:
- Parse Python source code into an abstract syntax tree (AST).
- Define a common interface for mutation operators.
- Change comparisons, Boolean operators, arithmetic operators, and boundaries.
- Remove or replace selected expressions while preserving syntactically valid Python.
- Record the mutation name, original code, mutated code, and affected location.
- Make mutations deterministic under a configured random seed.
- Verify that original code passes its tests before mutation.
- Retain a mutation only when the mutated program fails at least one relevant test.
- Prevent near-duplicate mutations from leaking across dataset splits.

Definitions:
- Mutation testing: introducing a small program change to determine whether tests detect it.
- Mutant: the deliberately modified version of a correct program.
- Mutation operator: a rule that creates one category of program defect.
- Abstract syntax tree (AST): a structured representation of source-code syntax.
- Equivalent mutant: a mutation that does not change observable program behavior.
- Kill a mutant: cause at least one test to fail because of the mutation.
- Data leakage: evaluation information appearing directly or indirectly in training data.
"""
