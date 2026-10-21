"""
Load a local model and generate code-repair predictions for evaluation tasks.

Topics to implement:
- Load a tokenizer and either a base model or fine-tuned checkpoint.
- Resolve CPU, MPS, and future CUDA devices explicitly.
- Apply the same prompt template used to construct training data.
- Implement deterministic greedy decoding as the initial evaluation policy.
- Add configurable sampling, temperature, top-p, and multiple candidates later.
- Decode only newly generated tokens rather than repeating the prompt.
- Extract Python from the model response without silently rewriting its behavior.
- Write task identifiers, generated code, model revision, and decoding settings to JSONL.
- Batch compatible requests and measure latency, throughput, and peak memory.

Definitions:
- Inference: using trained model parameters to produce predictions.
- Checkpoint: a saved set of model parameters and related metadata.
- Greedy decoding: selecting the highest-probability token at every generation step.
- Sampling: drawing tokens from the predicted probability distribution.
- Temperature: rescales token probabilities to control randomness.
- Top-p sampling: samples from the smallest token set reaching a probability threshold.
- Generation seed: initializes randomness so a sampling run can be reproduced.
- Throughput: amount of generated work, commonly tokens per second.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dataclasses import dataclass



class LocalModelRun:
    def __init__(self, model:AutoModelForCausalLM, tokenizer:AutoTokenizer, device:str):
        '''
        TODO: Potential breakpoint if we are fine tuning and model or tokenizer is not AutoModelForCausalLm or AutoTokenizer object respectively     
        '''
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @dataclass
    class GenerationResult:
        question: str
        reasoning: str
        answer: str
        raw_text: str
        completed_reasoning: bool

    
    def parse_completion(self, raw_text:str) -> tuple[str,str,bool]:
        reasoning, separator, answer = raw_text.partition("</think>")
        

        if not separator:
            return reasoning.strip(), "", False 

        return reasoning.strip(), answer.strip(), True
    
    @classmethod
    def from_pretrained(cls, model_name:str = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            padding_side="left") #adds left padding as LLM not trained to continue generation from padding tokens

        device = torch.device(
            'mps' if torch.mps.is_available() else 'cpu'
        )
        dtype = (
            torch.bfloat16 
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            low_cpu_mem_usage=True
            )
        model.to(device)
        model.eval()

        return cls(model, tokenizer, device)


    def generate(self,input_text:str) -> str:
        
        messages = [{
            "role": "user",
            "content": input_text,
            }]

        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors='pt',
            enable_thinking=True
        ).to(self.device)

        #model_inputs = self.tokenizer(messages, return_tensors="pt").to(self.model.device)
        
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens = 2048,
                do_sample=False)
        
        prompt_length = model_inputs['input_ids'].shape[-1]
        completion_ids = generated_ids[:, prompt_length:]
        
        model_output = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)[0]

        reasoning, answer, completed_reasoning = self.parse_completion(model_output)


        return self.GenerationResult(
            question = input_text,
            reasoning = reasoning.strip(),
            answer = answer.strip(),
            raw_text = model_output,
            completed_reasoning = completed_reasoning
        )

LocalModel = LocalModelRun.from_pretrained()
input_text = "give me a 5 word haiku about panthers"
generation_results = LocalModel.generate(input_text)
print('reasoning: ',generation_results.reasoning)
print('answer: ', generation_results.answer)



