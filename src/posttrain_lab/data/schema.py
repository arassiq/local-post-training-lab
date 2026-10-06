"""
Define and validate the data contracts used by training and evaluation.

Topics to implement:
- Define a supervised repair example with prompt inputs and known-correct target code.
- Define an evaluation task that contains hidden executable tests but no exposed answer.
- Represent train, validation, and test splits explicitly.
- Load newline-delimited JSON (JSONL) records from disk.
- Reject missing, unknown, empty, malformed, or duplicate fields.
- Format model prompts and target completions consistently.
- Preserve source, license, mutation, and repository provenance metadata.
- Add dataset fingerprints so an experiment can identify the exact input data.

Definitions:
- Data contract: the required structure and meaning of a record exchanged by components.
- JSONL: a text format containing one JSON object per line.
- Training example: an input paired with the target output used to calculate loss.
- Evaluation task: an input and hidden checks used to measure model behavior.
- Validation split: held-out data used for model and hyperparameter decisions.
- Test split: untouched data used only for the final performance estimate.
- Provenance: information describing where a record came from and how it was produced.
- Dataset fingerprint: a stable identifier derived from dataset contents and configuration.
"""
