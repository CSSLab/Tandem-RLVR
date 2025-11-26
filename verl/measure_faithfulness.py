#!/usr/bin/env python3
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from scipy import stats
from datetime import datetime
import os


def compute_token_entropy(logprobs_dict):
    if not logprobs_dict:
        return 0.0
    selected_logprob_obj = list(logprobs_dict.values())[0]
    if hasattr(selected_logprob_obj, 'logprob'):
        selected_logprob = selected_logprob_obj.logprob
    else:
        selected_logprob = float(selected_logprob_obj)
    entropy = -selected_logprob
    return entropy


def compute_sequence_entropy(token_logprobs):
    if not token_logprobs:
        return {'mean_entropy': 0.0, 'std_entropy': 0.0, 'total_tokens': 0}

    entropies = [compute_token_entropy(lp) for lp in token_logprobs if lp]
    if not entropies:
        return {'mean_entropy': 0.0, 'std_entropy': 0.0, 'total_tokens': 0}

    return {
        'mean_entropy': float(np.mean(entropies)),
        'std_entropy': float(np.std(entropies)),
        'max_entropy': float(np.max(entropies)),
        'min_entropy': float(np.min(entropies)),
        'median_entropy': float(np.median(entropies)),
        'total_tokens': len(entropies)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vanilla_model", type=str, default="/datadrive/difan/verl-llm-tandem/scratch/hf/Qwen3-0.6B-gsm8k-vanilla-step450")
    parser.add_argument("--tandem_model", type=str, default="/datadrive/difan/verl-llm-tandem/scratch/hf/Qwen3-0.6B-gsm8k-tandem-step60")
    parser.add_argument("--test_data", type=str, default="/datadrive/difan/verl-llm-tandem/scratch/data/gsm8k/test.parquet")
    parser.add_argument("--output_dir", type=str, default="/datadrive/difan/verl-llm-tandem/scratch/faithfulness")
    args = parser.parse_args()

    df = pd.read_parquet(args.test_data)
    raw_prompts = df['prompt'].tolist()
    prompts = [p[0]['content'] if isinstance(p, np.ndarray) and len(p) > 0 else str(p) for p in raw_prompts]
    num_samples = len(prompts)

    os.makedirs(args.output_dir, exist_ok=True)

    results = {
        'vanilla_model': args.vanilla_model,
        'tandem_model': args.tandem_model,
        'num_samples': num_samples,
        'models': {}
    }

    for model_name, model_path in [('vanilla', args.vanilla_model), ('tandem', args.tandem_model)]:
        print(f"\n{'='*80}")
        print(f"Measuring {model_name.upper()} model: {model_path}")
        print(f"{'='*80}")

        engine = LLM(
            model=model_path,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.85,
            trust_remote_code=True
        )

        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=1024,
            logprobs=1
        )

        print(f"Generating {num_samples} samples...")
        outputs = engine.generate(prompts, sampling_params)

        entropies = []
        for output in tqdm(outputs, desc=f"{model_name} entropy"):
            token_logprobs = output.outputs[0].logprobs
            entropy_stats = compute_sequence_entropy(token_logprobs)
            entropies.append(entropy_stats)

        results['models'][model_name] = {
            'mean_entropy': float(np.mean([e['mean_entropy'] for e in entropies])),
            'std_entropy': float(np.std([e['mean_entropy'] for e in entropies])),
            'median_entropy': float(np.median([e['mean_entropy'] for e in entropies])),
            'sample_entropies': entropies
        }

        print(f"✓ {model_name} mean entropy: {results['models'][model_name]['mean_entropy']:.4f}")

        del engine

    vanilla_entropies = [e['mean_entropy'] for e in results['models']['vanilla']['sample_entropies']]
    tandem_entropies = [e['mean_entropy'] for e in results['models']['tandem']['sample_entropies']]

    t_stat, p_value = stats.ttest_ind(tandem_entropies, vanilla_entropies)

    results['comparison'] = {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'entropy_diff': float(np.mean(tandem_entropies) - np.mean(vanilla_entropies)),
        'significant': p_value < 0.05
    }

    print(f"\n{'='*80}")
    print(f"COMPARISON")
    print(f"{'='*80}")
    print(f"Vanilla mean entropy: {results['models']['vanilla']['mean_entropy']:.4f}")
    print(f"Tandem mean entropy:  {results['models']['tandem']['mean_entropy']:.4f}")
    print(f"Difference (tandem - vanilla): {results['comparison']['entropy_diff']:.4f}")
    print(f"t-statistic: {results['comparison']['t_statistic']:.4f}")
    print(f"p-value: {results['comparison']['p_value']:.4f}")
    print(f"Significant: {results['comparison']['significant']}")

    if results['comparison']['entropy_diff'] > 0:
        print(f"\n→ Tandem model has HIGHER entropy (LESS faithful reasoning)")
    else:
        print(f"\n→ Tandem model has LOWER entropy (MORE faithful reasoning)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"faithfulness_{timestamp}.json")

    results_to_save = {k: v for k, v in results.items()}
    for model_key in results_to_save['models']:
        del results_to_save['models'][model_key]['sample_entropies']

    with open(output_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
