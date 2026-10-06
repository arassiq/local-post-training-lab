# Dataset contract

## Supervised example

Required fields:

- `id`: stable unique identifier
- `instruction`: requested repair behavior
- `buggy_code`: input implementation
- `test_output`: diagnostic information visible to the model
- `target_code`: known-good completion
- `source`: provenance label
- `split`: `train` or `validation`

## Functional task

Required fields:

- `id`: stable unique identifier
- `instruction`: requested behavior
- `buggy_code`: starting implementation
- `test_code`: executable hidden tests
- `source`: provenance label
- `split`: `validation` or `test`

## Mutation pipeline

Mutation-generated examples must satisfy both conditions:

1. The original implementation passes every test.
2. The mutated implementation fails at least one test.

Split source repositories before applying mutations. Randomly splitting mutations
after generation can leak nearly identical implementations across train and test.

Never redistribute source code or model-generated derivatives without checking
the source license and attribution requirements.
