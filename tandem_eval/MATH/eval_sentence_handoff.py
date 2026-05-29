import sys
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "verl"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "tandem_eval"))

import json
import numpy as np
import torch
import pandas as pd
from scipy.special import comb
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from grader import load_grader

SCRATCH = os.environ.get("TANDEM_SCRATCH", os.path.join(_REPO_ROOT, "scratch"))
BENCHMARKS = {
    'amc_23_25':  f'{SCRATCH}/MATH/amc_23_25/test.parquet',
    'aime_24_26': f'{SCRATCH}/MATH/aime_24_26/test.parquet',
    'minerva':    f'{SCRATCH}/MATH/minerva/test.parquet',
}
JUNIOR_MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
MAX_MODEL_LEN = 4096
MAX_TOKENS = 3000
N_SAMPLES = 4
K_VALUES = (1, 4)

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

GEN_DIR = os.path.join(os.path.dirname(__file__), 'generations')


def derive_sentence_attribution(mask, token_ids, boundary_set):
    sentences, current_model, current_len = [], None, 0
    for idx, (m, tid) in enumerate(zip(mask, token_ids)):
        label = 'senior' if m == 1 else 'junior'
        if current_model is None:
            current_model = label
        current_len += 1
        if tid in boundary_set or idx == len(mask) - 1:
            sentences.append((current_model, current_len))
            current_model, current_len = None, 0
    return sentences


def eval_sentence_handoff(senior_path, senior_name, val_file=None, benchmark_name='math500',
                           max_samples=None):
    val_file = val_file or BENCHMARKS['amc_23_25']
    tokenizer = AutoTokenizer.from_pretrained(senior_path, trust_remote_code=True)

    boundary_token_ids = sorted({tid for tid in range(tokenizer.vocab_size)
                                  if tokenizer.decode([tid]).endswith('\n\n')})
    print(f"  Boundary: paragraph (\\n\\n-ending tokens, n={len(boundary_token_ids)}, "
          f"e.g. {boundary_token_ids[:5]})")

    df = pd.read_parquet(val_file)
    if max_samples is not None:
        df = df.head(max_samples)

    prompts, ground_truths = [], []
    for _, row in df.iterrows():
        prompt = row['prompt']
        if hasattr(prompt, 'tolist'):
            prompt = prompt.tolist()
        formatted = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True)
        prompts.append(str(formatted))
        ground_truths.append(str(row['reward_model']['ground_truth']))

    llm = LLM(
        model=senior_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        dtype='bfloat16',
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
        tandem_config={
            'enabled': True,
            'frozen_model': JUNIOR_MODEL,
            'selection_strategy': 'sentence',
            'prob_primary': 0.5,
            'boundary_token_ids': boundary_token_ids,
            'frozen_gpu_devices': [1],
        },
    )

    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, n=N_SAMPLES,
                                     max_tokens=MAX_TOKENS)

    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    boundary_set = set(boundary_token_ids)
    correct_counts, generations = [], []

    for i, output in enumerate(outputs):
        samp_scores, samp_responses, samp_masks, samp_sr_tfrac, samp_sr_sfrac, samp_n_sents = [], [], [], [], [], []
        for samp in output.outputs:
            response = samp.text
            score = score_fn(response, ground_truths[i])
            samp_scores.append(score)
            samp_responses.append(response)
            mask = samp.tandem_model_mask or []
            token_ids = samp.token_ids if hasattr(samp, 'token_ids') else []
            sent_attr = derive_sentence_attribution(mask, token_ids, boundary_set) if mask and token_ids else []
            n_senior_toks = sum(mask)
            n_total_toks = len(mask)
            n_senior_sents = sum(1 for lbl, _ in sent_attr if lbl == 'senior')
            n_total_sents = len(sent_attr)
            samp_masks.append(list(mask))
            samp_sr_tfrac.append(n_senior_toks / max(n_total_toks, 1))
            samp_sr_sfrac.append(n_senior_sents / max(n_total_sents, 1))
            samp_n_sents.append(n_total_sents)
        c = sum(1 for s in samp_scores if s >= 1.0)
        correct_counts.append(c)
        generations.append({
            'idx': i,
            'ground_truth': ground_truths[i],
            'sample_scores': samp_scores,
            'correct_count': c,
            'n_samples': len(samp_scores),
            'responses': samp_responses,
            'senior_token_frac_per_sample': samp_sr_tfrac,
            'senior_sent_frac_per_sample': samp_sr_sfrac,
            'n_sentences_per_sample': samp_n_sents,
            'token_masks': samp_masks,
        })

    total = len(correct_counts)
    pass_at_k = compute_pass_at_k(correct_counts, N_SAMPLES, K_VALUES)
    all_sr_tfrac = np.concatenate([g['senior_token_frac_per_sample'] for g in generations])
    all_sr_sfrac = np.concatenate([g['senior_sent_frac_per_sample'] for g in generations])
    all_n_sents  = np.concatenate([g['n_sentences_per_sample'] for g in generations])
    avg_sents    = float(np.mean(all_n_sents))
    avg_sr_tfrac = float(np.mean(all_sr_tfrac))
    avg_sr_sfrac = float(np.mean(all_sr_sfrac))

    print(f"\n  Sentence Handoff ({senior_name})")
    pass_str = ' | '.join(f'pass@{k}: {v:5.1f}%' for k, v in pass_at_k.items())
    print(f"  {pass_str} | N={total}")
    print(f"  Avg lines: {avg_sents:.1f} | Senior line%: {100*avg_sr_sfrac:.1f}% "
          f"| Senior tok%: {100*avg_sr_tfrac:.1f}%")

    os.makedirs(GEN_DIR, exist_ok=True)
    slug = senior_name.lower().replace(' ', '_')
    with open(os.path.join(GEN_DIR, f'sentence_handoff_{slug}_{benchmark_name}.jsonl'), 'w') as f:
        for g in generations:
            f.write(json.dumps(g, ensure_ascii=False) + '\n')

    del llm
    torch.cuda.empty_cache()

    return {
        'model_name': senior_name,
        'pass_at_k': pass_at_k,
        'n_samples': N_SAMPLES,
        'total': total,
        'avg_sentences': avg_sents,
        'avg_senior_sent_frac': avg_sr_sfrac,
        'avg_senior_token_frac': avg_sr_tfrac,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--model-name', default=None)
    args = parser.parse_args()

    if args.model_path and args.model_name:
        MODELS = [(args.model_path, args.model_name)]
    else:
        MODELS = [
            (f'{SCRATCH}/hf/tandem_word_grpo_deepscaler_at4_Qwen3-4B-Instruct-2507_gap32_no-jr-tkn-weight_step120/best_model', 'TT-Word-4B-DSR'),
            (f'{SCRATCH}/hf/vanilla_grpo_deepscaler_Qwen3-4B-Instruct-2507/best_model',                                       'Vanilla-GRPO-4B-DSR'),
        ]

    print('=' * 70)
    print(f'SENTENCE HANDOFF  Junior={JUNIOR_MODEL} | Boundary=newline | n={N_SAMPLES}')
    print('=' * 70)

    all_results = {}
    for senior_path, senior_name in MODELS:
        for bname, bfile in BENCHMARKS.items():
            print(f'\n--- {senior_name} | {bname.upper()} ---')
            all_results[(senior_name, bname)] = eval_sentence_handoff(
                senior_path, senior_name, val_file=bfile, benchmark_name=bname)

    print('\n' + '=' * 70)
    print(f'SUMMARY (n={N_SAMPLES})')
    print('=' * 70)
    print("{:<20} {:<12} {:>8} {:>8} {:>8} {:>9} {:>8}".format(
        'Model', 'Benchmark', 'pass@1', 'pass@4', '#Lines', 'SrLine%', 'SrTok%'))
    print('-' * 80)
    for (sname, bname), r in all_results.items():
        pa = r['pass_at_k']
        print("{:<20} {:<12} {:>7.1f}% {:>7.1f}% {:>8.1f} {:>8.1f}% {:>7.1f}%".format(
            sname, bname, pa.get(1, float('nan')), pa.get(4, float('nan')),
            r['avg_sentences'], 100 * r['avg_senior_sent_frac'],
            100 * r['avg_senior_token_frac']))
    print('=' * 70)
