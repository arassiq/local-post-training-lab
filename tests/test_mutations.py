import ast

from posttrain_lab.data.mutations import FlipBoolean, FlipComparison


def test_flip_comparison_changes_first_operator() -> None:
    mutation = FlipComparison().apply("def eligible(age):\n    return age >= 18\n")
    assert mutation is not None
    assert "age > 18" in mutation.mutated_code
    ast.parse(mutation.mutated_code)


def test_flip_boolean_changes_first_operator() -> None:
    mutation = FlipBoolean().apply("def inside(x):\n    return x > 0 and x < 10\n")
    assert mutation is not None
    assert " or " in mutation.mutated_code
    ast.parse(mutation.mutated_code)


def test_operator_returns_none_when_not_applicable() -> None:
    assert FlipComparison().apply("def identity(x):\n    return x\n") is None
