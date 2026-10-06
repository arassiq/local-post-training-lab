from posttrain_lab.inference.generate import extract_python


def test_extracts_fenced_python() -> None:
    completion = "```python\ndef add(a, b):\n    return a + b\n```\nExplanation"
    assert extract_python(completion) == "def add(a, b):\n    return a + b\n"


def test_preserves_unfenced_code() -> None:
    assert extract_python("def identity(x):\n    return x") == "def identity(x):\n    return x\n"
