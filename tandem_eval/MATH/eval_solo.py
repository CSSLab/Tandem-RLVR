import sys
import os
sys.path.insert(0, '/datadrive/difan/verl-llm-tandem/verl')
sys.path.insert(0, '/datadrive/difan/verl-llm-tandem/tandem_eval')

import json
import numpy as np
import torch
import pandas as pd
from scipy.special import comb
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from grader import load_grader

SCRATCH = '/datadrive/difan/verl-llm-tandem/scratch'

BENCHMARKS = {
    'amc_23_25':  f'{SCRATCH}/MATH/amc_23_25/test.parquet',
    'aime_24_26': f'{SCRATCH}/MATH/aime_24_26/test.parquet',
    'minerva':    f'{SCRATCH}/MATH/minerva/test.parquet',
    'math500':    f'{SCRATCH}/MATH/math500/test.parquet',
}

MAX_MODEL_LEN = 4096
MAX_TOKENS    = 3000
N_SAMPLES     = 16
K_VALUES      = (1, 2, 4, 8, 16, 32)

score_fn = load_grader('math')


def compute_pass_at_k(correct_counts, n, k_values=K_VALUES):
    out = {}
    for k in k_values:
        if k > n:
            continue
        vals = []
        for c in correct_counts:
            if n - c >= k:
                vals.append(1.0 - comb(n - c, k, exact=False) / comb(n, k, exact=False))
            else:
                vals.append(1.0)
        out[k] = float(np.mean(vals)) * 100
    return out


def eval_model_on_benchmark(llm, tokenizer, benchmark_name, parquet_path):
    df = pd.read_parquet(parquet_path)

    prompts, ground_truths = [], []
    for _, row in df.iterrows():
        prompt = row['prompt']
        if hasattr(prompt, 'tolist'):
            prompt = prompt.tolist()
        formatted = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        prompts.append(str(formatted))
        ground_truths.append(str(row['reward_model']['ground_truth']))

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        n=N_SAMPLES,
        max_tokens=MAX_TOKENS,
        detokenize=False,
    )

    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    correct_counts, generations = [], []
    for i, output in enumerate(outputs):
        samp_scores, samp_responses, samp_ntoks = [], [], []
        for samp in output.outputs:
            resp = tokenizer.decode(samp.token_ids, skip_special_tokens=True)
            s = score_fn(resp, ground_truths[i])
            samp_scores.append(s)
            samp_responses.append(resp)
            samp_ntoks.append(len(samp.token_ids))
        c = sum(1 for s in samp_scores if s >= 1.0)
        correct_counts.append(c)
        generations.append({
            'idx': i,
            'ground_truth': ground_truths[i],
            'sample_scores': samp_scores,
            'correct_count': c,
            'n_samples': len(samp_scores),
            'responses': samp_responses,
            'n_tokens': samp_ntoks,
        })

    total = len(correct_counts)
    pass_at_k = compute_pass_at_k(correct_counts, N_SAMPLES, K_VALUES)
    pass_str = ' | '.join(f'pass@{k}: {v:5.1f}%' for k, v in pass_at_k.items())
    print(f"  {benchmark_name:<12} | {pass_str} | N={total}")
    return {'benchmark': benchmark_name, 'pass_at_k': pass_at_k,
            'n_samples': N_SAMPLES, 'total': total, 'generations': generations}


def eval_model(model_path, model_name, benchmarks=None, save_generations=True):
    if benchmarks is None:
        benchmarks = list(BENCHMARKS.keys())

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Path:  {model_path}")
    print(f"{'='*60}")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        dtype='bfloat16',
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    results = []
    for bname in benchmarks:
        result = eval_model_on_benchmark(llm, tokenizer, bname, BENCHMARKS[bname])
        result['model_name'] = model_name
        results.append(result)

        if save_generations:
            gen_dir = os.path.join(os.path.dirname(__file__), 'generations')
            os.makedirs(gen_dir, exist_ok=True)
            slug = model_name.lower().replace(' ', '_').replace('/', '_')
            gen_file = os.path.join(gen_dir, f'{slug}_{bname}.jsonl')
            with open(gen_file, 'w') as f:
                for g in result['generations']:
                    f.write(json.dumps(g, ensure_ascii=False) + '\n')

    del llm
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--model-name', default=None)
    parser.add_argument('--n-samples', type=int, default=None,
                        help=f'Override N_SAMPLES (default {N_SAMPLES}). Use 1 for quick sanity.')
    args = parser.parse_args()
    if args.n_samples is not None:
        N_SAMPLES = args.n_samples
        K_VALUES = tuple(k for k in (1, 2, 4, 8, 16, 32) if k <= N_SAMPLES)
        if not K_VALUES:
            K_VALUES = (1,)

    if args.model_path and args.model_name:
        models = [(args.model_path, args.model_name)]
    else:
        models = [
            ('Qwen/Qwen3-0.6B-Base', 'Qwen3-0.6B-Base'),
            # (f'{SCRATCH}/hf/tandem_word_grpo_deepscaler_at4_Qwen3-4B-Instruct-2507_gap32_no-jr-tkn-weight_step120/best_model', 'TT-Word-4B-DSR'),
            # (f'{SCRATCH}/hf/vanilla_grpo_deepscaler_Qwen3-4B-Instruct-2507/best_model',                                       'Vanilla-GRPO-4B-DSR'),
        ]

    all_results = {}
    for model_path, model_name in models:
        results = eval_model(model_path, model_name)
        all_results[model_name] = {r['benchmark']: r for r in results}

    print(f"\n{'='*80}")
    print(f'SUMMARY  (n={N_SAMPLES}; reporting pass@k)')
    print(f"{'='*80}")
    header = f"{'Model':<25}" + ''.join(f"  {b:<22}" for b in BENCHMARKS)
    print(header)
    print('-' * len(header))
    for model_name, bdict in all_results.items():
        row = f"{model_name:<25}"
        for bname in BENCHMARKS:
            if bname in bdict:
                pa = bdict[bname]['pass_at_k']
                cell = ' / '.join(f'@{k}={v:5.1f}' for k, v in pa.items())
                row += f"  {cell:<22}"
            else:
                row += f"  {'nan':<22}"
        print(row)
    print('=' * 80)
