"""
Test the training and evaluation data contracts.

Topics to implement:
- Accept complete valid training examples and evaluation tasks.
- Reject malformed JSON, missing fields, unknown fields, and empty required values.
- Reject duplicate identifiers within one dataset.
- Reject invalid split names and training examples placed in evaluation splits.
- Verify prompt and completion formatting exactly.
- Verify that dataset fingerprints are stable and change when content changes.

Definitions:
- Unit test: a focused check of one small behavior in isolation.
- Fixture: reusable test input or setup state.
- Boundary case: input at the edge of the valid or invalid domain.
"""
