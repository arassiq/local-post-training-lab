from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from posttrain_lab.data.schema import RepairExample, load_jsonl


@dataclass(frozen=True)
class EncodedExample:
    input_ids: list[int]
    labels: list[int]


def encode_example(example: RepairExample, tokenizer: Any, max_length: int) -> EncodedExample:
    prompt_ids = tokenizer.encode(example.prompt(), add_special_tokens=True)
    completion_ids = tokenizer.encode(example.completion(), add_special_tokens=False)
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        completion_ids = [*completion_ids, eos_id]
    input_ids = [*prompt_ids, *completion_ids][:max_length]
    prompt_length = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prompt_length + input_ids[prompt_length:]
    if not any(label != -100 for label in labels):
        raise ValueError(f"{example.id}: target was completely truncated; increase max_length")
    return EncodedExample(input_ids=input_ids, labels=labels)


class RepairDataset:
    def __init__(self, examples: list[RepairExample], tokenizer: Any, max_length: int) -> None:
        self.items = [encode_example(example, tokenizer, max_length) for example in examples]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> EncodedExample:
        return self.items[index]


class CompletionCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, items: list[EncodedExample]) -> dict[str, Any]:
        import torch

        width = max(len(item.input_ids) for item in items)
        input_ids = []
        labels = []
        attention_mask = []
        for item in items:
            padding = width - len(item.input_ids)
            input_ids.append(item.input_ids + [self.pad_token_id] * padding)
            labels.append(item.labels + [-100] * padding)
            attention_mask.append([1] * len(item.input_ids) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def run_training(config_path: Path) -> None:
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    model_name = config["model"]["name"]
    revision = config["model"]["revision"]
    trust_remote_code = bool(config["model"]["trust_remote_code"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    precision = config["training"]["precision"]
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
    ).to(device)
    if config["training"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    max_length = int(config["data"]["max_length"])
    train_examples = load_jsonl(Path(config["data"]["train_path"]), RepairExample)
    validation_examples = load_jsonl(Path(config["data"]["validation_path"]), RepairExample)
    train_dataset = RepairDataset(train_examples, tokenizer, max_length)
    validation_dataset = RepairDataset(validation_examples, tokenizer, max_length)
    collator = CompletionCollator(tokenizer.pad_token_id)
    batch_size = int(config["training"]["micro_batch_size"])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator
    )

    optimizer = torch.optim.AdamW(
        _parameter_groups(model, float(config["training"]["weight_decay"])),
        lr=float(config["training"]["learning_rate"]),
        betas=(
            float(config["training"]["adam_beta1"]),
            float(config["training"]["adam_beta2"]),
        ),
        eps=float(config["training"]["adam_epsilon"]),
    )
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    configured_max = int(config["training"]["max_steps"])
    total_steps = min(configured_max, steps_per_epoch * int(config["training"]["epochs"]))
    warmup_steps = round(total_steps * float(config["training"]["warmup_ratio"]))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    state = {"step": 0, "examples_seen": 0}
    optimizer.zero_grad(set_to_none=True)
    model.train()
    for _epoch in range(int(config["training"]["epochs"])):
        for micro_step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss / accumulation
            loss.backward()
            state["examples_seen"] += batch["input_ids"].shape[0]
            if micro_step % accumulation != 0 and micro_step != len(train_loader):
                continue
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["max_grad_norm"])
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state["step"] += 1

            if state["step"] % int(config["training"]["log_every"]) == 0:
                print(json.dumps({**state, "train_loss": float(loss.item() * accumulation)}))
            if state["step"] % int(config["training"]["eval_every"]) == 0:
                validation_loss = _validation_loss(model, validation_loader, device)
                print(json.dumps({**state, "validation_loss": validation_loss}))
                model.train()
            if state["step"] % int(config["training"]["save_every"]) == 0:
                _save_checkpoint(model, tokenizer, optimizer, scheduler, output_dir, state)
            if state["step"] >= total_steps:
                break
        if state["step"] >= total_steps:
            break

    _save_checkpoint(model, tokenizer, optimizer, scheduler, output_dir, state, final=True)


def _parameter_groups(model: Any, weight_decay: float) -> list[dict[str, Any]]:
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith("bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _validation_loss(model: Any, loader: Any, device: Any) -> float:
    import torch

    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            losses.append(float(model(**batch).loss.item()))
    return sum(losses) / len(losses)


def _save_checkpoint(
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    output_dir: Path,
    state: dict[str, int],
    *,
    final: bool = False,
) -> None:
    import torch

    name = "final" if final else f"checkpoint-{state['step']:06d}"
    checkpoint = output_dir / name
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "state": state},
        checkpoint / "trainer_state.pt",
    )


def _validate_config(config: dict[str, Any]) -> None:
    required = {"experiment", "model", "data", "training"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"configuration missing sections: {sorted(missing)}")
    if config["model"]["name"] == "REPLACE_ME":
        raise ValueError("set model.name in the configuration before training")
