import sys
sys.path.insert(0, '/datadrive/difan/verl-llm-tandem/verl')

import torch
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from verl.utils.reward_score.gsm8k import compute_score

model_path = "Qwen/Qwen3-0.6B"
cache_dir = "/datadrive/difan/verl-llm-tandem/scratch/models"
val_file = "/datadrive/difan/verl-llm-tandem/scratch/data/gsm8k/test.parquet"

print(f"Loading model: {model_path}")
llm = LLM(
    model=model_path,
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9,
    trust_remote_code=True,
    dtype='bfloat16',
    enforce_eager=True,
    max_model_len=512 + 1024,
    download_dir=cache_dir,
)

tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir, trust_remote_code=True)

sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    max_tokens=1024,
    detokenize=False,
)

print(f"Loading validation data: {val_file}")
df = pd.read_parquet(val_file)
print(f"Total samples: {len(df)}")

prompts = []
ground_truths = []
for idx, row in df.iterrows():
    prompt = row['prompt']
    if isinstance(prompt, list):
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    ground_truth = row['reward_model']['ground_truth']
    prompts.append(str(prompt))
    ground_truths.append(ground_truth)

print(f"Generating responses...")
outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

print(f"\nEvaluating...")
scores = []
for i, output in enumerate(outputs):
    response = tokenizer.decode(output.outputs[0].token_ids, skip_special_tokens=True)
    score = compute_score(response, str(ground_truths[i]), method="strict", format_score=0.1, score=1.0)
    scores.append(score)

    if i < 2:
        print(f"\n[Sample {i}]")
        print(f"Prompt: {prompts[i][:100]}...")
        print(f"Response: {response[:200]}...")
        print(f"Ground truth: {ground_truths[i]}")
        print(f"Score: {score}")

avg_score = sum(scores) / len(scores)
print(f"\n{'='*80}")
print(f"Base Model Evaluation: {model_path}")
print(f"Average Score: {avg_score:.4f}")
print(f"Num samples: {len(scores)}")
print(f"{'='*80}")
