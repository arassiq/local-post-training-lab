from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from posttrain_lab.data.schema import RepairTask, load_jsonl


def generate_predictions(
    model_name: str,
    task_path: Path,
    output_path: Path,
    *,
    revision: str = "main",
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    seed: int = 17,
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    tasks = load_jsonl(task_path, RepairTask)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, torch_dtype=dtype
    ).to(device)
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            inputs = tokenizer(task.prompt(), return_tensors="pt").to(device)
            input_length = inputs["input_ids"].shape[1]
            generation_options: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if temperature > 0:
                generation_options.update(do_sample=True, temperature=temperature)
            else:
                generation_options["do_sample"] = False
            with torch.no_grad():
                output = model.generate(**inputs, **generation_options)
            completion = tokenizer.decode(output[0, input_length:], skip_special_tokens=True)
            record = {"task_id": task.id, "generated_code": extract_python(completion)}
            handle.write(json.dumps(record) + "\n")


def extract_python(completion: str) -> str:
    """Remove the markdown fence used by the training format without rewriting code."""
    value = completion.strip()
    if value.startswith("```python"):
        value = value[len("```python") :].lstrip("\n")
    elif value.startswith("```"):
        value = value[3:].lstrip("\n")
    if "```" in value:
        value = value.split("```", maxsplit=1)[0]
    return value.rstrip() + "\n"
