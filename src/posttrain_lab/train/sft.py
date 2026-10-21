"""
Handle supervised fine-tuning (SFT) for a causal language model.

Topics to implement:
- Load the experiment configuration, tokenizer, and base model.
- Select the local compute device, especially Apple Metal Performance Shaders (MPS).
- Convert repair examples into prompt and target token sequences.
- Mask prompt and padding tokens so loss is calculated only on target tokens.
- Build a dataset, collator, and shuffled training data loader.
- Configure full-parameter AdamW with appropriate weight-decay parameter groups.
- Implement gradient accumulation, gradient clipping, and a learning-rate schedule.
- Add validation without gradient updates.
- Save model, tokenizer, optimizer, scheduler, and trainer state checkpoints.
- Record reproducibility information such as seed, model revision, and configuration.

Definitions:
- Supervised fine-tuning (SFT): training on input and known-correct output pairs.
- Teacher forcing: predicting each target token while receiving previous correct tokens.
- Cross-entropy loss: measures how much probability the model assigns to target tokens.
- Label masking: setting ignored labels to -100 so they do not contribute to loss.
- AdamW: Adam optimization with decoupled weight decay.
- Micro-batch: examples processed in one forward and backward pass.
- Effective batch: micro-batch size multiplied by gradient-accumulation steps.
- Gradient accumulation: delaying an optimizer update across multiple micro-batches.
- Gradient clipping: limiting gradient magnitude to reduce unstable updates.
- Checkpoint: saved training state that can be evaluated or resumed later.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=16384
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

content = tokenizer.decode(output_ids, skip_special_tokens=True)
