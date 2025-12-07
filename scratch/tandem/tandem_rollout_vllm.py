import torch
from vllm import LLM, SamplingParams
from tensordict import TensorDict
from verl import DataProto
from verl.utils.torch_functional import get_response_mask
from concurrent.futures import ThreadPoolExecutor
import os

class TandemRolloutVLLM:
    def __init__(self, model_a_path, model_b_path, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.prob_a = config.get("prob_a", 0.5)

        gpu_memory_util = config.get("gpu_memory_utilization", 0.45)
        dtype = config.get("dtype", "bfloat16")
        gpu_a = config.get("gpu_a", 0)
        gpu_b = config.get("gpu_b", 1)

        from vllm import LLM
        import torch

        print(f"[TandemRolloutVLLM] Initializing hot model on cuda:{gpu_a}: {model_a_path}")
        with torch.cuda.device(gpu_a):
            self.vllm_a = LLM(
                model=model_a_path,
                tensor_parallel_size=1,
                gpu_memory_utilization=gpu_memory_util,
                dtype=dtype,
                enforce_eager=True,
                max_model_len=config.get("max_model_len", 2048),
                enable_prefix_caching=True,
                device=f"cuda:{gpu_a}"
            )

        print(f"[TandemRolloutVLLM] Initializing frozen model on cuda:{gpu_b}: {model_b_path}")
        with torch.cuda.device(gpu_b):
            self.vllm_b = LLM(
                model=model_b_path,
                tensor_parallel_size=1,
                gpu_memory_utilization=gpu_memory_util,
                dtype=dtype,
                enforce_eager=True,
                max_model_len=config.get("max_model_len", 2048),
                enable_prefix_caching=True,
                device=f"cuda:{gpu_b}"
            )

        print(f"[TandemRolloutVLLM] Both models initialized successfully")

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // self.config.get("micro_batch_size", batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        temperature = prompts.meta_info.get("temperature", self.config.get("temperature", 0.7))
        response_length = prompts.meta_info.get("response_length", self.config.get("response_length", 64))
        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 0.95))

        input_ids = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        batch_size = input_ids.size(0)
        prompt_length = input_ids.size(1)
        max_new_tokens = response_length

        device = torch.device("cuda:0")

        current_token_ids = [
            input_ids[i][attention_mask[i].bool()].tolist()
            for i in range(batch_size)
        ]

        response_tokens = [[] for _ in range(batch_size)]
        model_a_mask = torch.zeros((batch_size, max_new_tokens), dtype=torch.bool, device=device)
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)

        vocab_size = len(self.tokenizer)

        from vllm import SamplingParams
        from concurrent.futures import ThreadPoolExecutor

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=20,
            prompt_logprobs=0
        )

        executor = ThreadPoolExecutor(max_workers=2)

        for step in range(max_new_tokens):
            if done.all():
                break

            future_a = executor.submit(self.vllm_a.generate, prompt_token_ids=current_token_ids, sampling_params=sampling_params, use_tqdm=False)
            future_b = executor.submit(self.vllm_b.generate, prompt_token_ids=current_token_ids, sampling_params=sampling_params, use_tqdm=False)

            outputs_a = future_a.result()
            outputs_b = future_b.result()

            logits_a = torch.full((batch_size, vocab_size), -100.0, device=device, dtype=torch.float32)
            logits_b = torch.full((batch_size, vocab_size), -100.0, device=device, dtype=torch.float32)

            for i in range(batch_size):
                if done[i]:
                    continue

                output_a = outputs_a[i].outputs[0]
                output_b = outputs_b[i].outputs[0]

                if output_a.logprobs and len(output_a.logprobs) > 0:
                    for token_id, logprob_obj in output_a.logprobs[0].items():
                        logits_a[i, token_id] = logprob_obj.logprob

                if output_b.logprobs and len(output_b.logprobs) > 0:
                    for token_id, logprob_obj in output_b.logprobs[0].items():
                        logits_b[i, token_id] = logprob_obj.logprob

            candidate_a = self._sample_sparse(logits_a, temperature)
            candidate_b = self._sample_sparse(logits_b, temperature)

            is_turn_a = torch.rand(batch_size, device=device) < self.prob_a
            chosen_token = torch.where(is_turn_a, candidate_a, candidate_b)

            for i in range(batch_size):
                if not done[i]:
                    token = chosen_token[i].item()
                    response_tokens[i].append(token)
                    current_token_ids[i].append(token)
                    model_a_mask[i, step] = is_turn_a[i]

                    if token == eos_token_id:
                        done[i] = True

        response = torch.full((batch_size, max_new_tokens), pad_token_id, dtype=torch.long, device=device)
        for i in range(batch_size):
            resp_len = min(len(response_tokens[i]), max_new_tokens)
            response[i, :resp_len] = torch.tensor(response_tokens[i][:resp_len], dtype=torch.long, device=device)

        prompt = input_ids.to(device)
        seq = torch.cat([prompt, response], dim=1)

        delta_position_id = torch.arange(1, max_new_tokens + 1, device=device).unsqueeze(0).repeat(batch_size, 1)
        response_position_ids = position_ids.to(device)[:, -1:] + delta_position_id
        position_ids_full = torch.cat([position_ids.to(device), response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(
            response_id=response,
            eos_token=eos_token_id,
            dtype=attention_mask.dtype
        )
        attention_mask_full = torch.cat([attention_mask.to(device), response_attention_mask], dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask_full,
                "position_ids": position_ids_full,
                "model_a_mask": model_a_mask.cpu(),
            },
            batch_size=batch_size,
        )

        return DataProto(batch=batch)

    def _sample_sparse(self, logits, temperature):
        if temperature < 1e-5:
            return logits.argmax(dim=-1)
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return sampled

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import time
    from tandem_rollout import TandemRollout

    model_path = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_prompts = [
        "The capital of France is",
        "In mathematics, a prime number is"
    ]

    response_length = 256

    input_ids_list = []
    attention_mask_list = []
    for prompt in test_prompts:
        encoded = tokenizer(prompt, return_tensors="pt", padding=False)
        input_ids_list.append(encoded["input_ids"][0])
        attention_mask_list.append(encoded["attention_mask"][0])

    max_len = max(len(ids) for ids in input_ids_list)
    batch_input_ids = torch.full((len(test_prompts), max_len), tokenizer.pad_token_id, dtype=torch.long)
    batch_attention_mask = torch.zeros((len(test_prompts), max_len), dtype=torch.long)

    for i, (ids, mask) in enumerate(zip(input_ids_list, attention_mask_list)):
        batch_input_ids[i, :len(ids)] = ids
        batch_attention_mask[i, :len(mask)] = mask

    position_ids = batch_attention_mask.cumsum(dim=1) - 1
    position_ids = position_ids.masked_fill(batch_attention_mask == 0, 0)

    batch_dict = TensorDict({
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "position_ids": position_ids
    }, batch_size=len(test_prompts))

    meta_info = {
        "temperature": 0.7,
        "response_length": response_length,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "top_p": 0.95
    }

    prompts_proto = DataProto(batch=batch_dict, meta_info=meta_info)

    print("="*80)
    print("BENCHMARK: vLLM Tandem vs HuggingFace Tandem")
    print(f"Generating {response_length} tokens with 2 prompts")
    print("="*80)

    print("\n[1/2] Testing vLLM Tandem Rollout...")
    config_vllm = {
        "prob_a": 0.5,
        "temperature": 0.7,
        "response_length": response_length,
        "micro_batch_size": 2,
        "gpu_memory_utilization": 0.45,
        "dtype": "bfloat16",
        "max_model_len": 2048,
        "top_p": 0.95,
        "gpu_a": 0,
        "gpu_b": 1
    }

    tandem_vllm = TandemRolloutVLLM(
        model_a_path=model_path,
        model_b_path=model_path,
        tokenizer=tokenizer,
        config=config_vllm
    )

    start = time.time()
    output_vllm = tandem_vllm.generate_sequences(prompts_proto)
    elapsed_vllm = time.time() - start

    print(f"vLLM Tandem took: {elapsed_vllm:.2f}s ({elapsed_vllm/response_length:.4f}s per token)")

    print("\n[2/2] Testing HuggingFace Tandem Rollout...")
    print("Loading HuggingFace models...")
    model_a_hf = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    )
    model_b_hf = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:1"
    )

    config_hf = {
        "prob_a": 0.5,
        "temperature": 0.7,
        "response_length": response_length,
        "micro_batch_size": 2,
        "top_p": 0.95
    }

    tandem_hf = TandemRollout(
        model_a=model_a_hf,
        model_b=model_b_hf,
        tokenizer=tokenizer,
        config=config_hf
    )

    start = time.time()
    output_hf = tandem_hf.generate_sequences(prompts_proto)
    elapsed_hf = time.time() - start

    print(f"HuggingFace Tandem took: {elapsed_hf:.2f}s ({elapsed_hf/response_length:.4f}s per token)")

    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(f"vLLM Tandem:        {elapsed_vllm:.2f}s  ({elapsed_vllm/response_length:.4f}s/token)")
    print(f"HuggingFace Tandem: {elapsed_hf:.2f}s  ({elapsed_hf/response_length:.4f}s/token)")
    print(f"Speedup:            {elapsed_hf/elapsed_vllm:.2f}x faster with vLLM")
    print("="*80)

    print("\nvLLM Output Sample:")
    for i in range(len(test_prompts)):
        response_ids = output_vllm.batch["responses"][i]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        model_a_tokens = output_vllm.batch["model_a_mask"][i].sum().item()
        total_tokens = (~(response_ids == tokenizer.pad_token_id)).sum().item()
        print(f"\nPrompt {i+1}: {test_prompts[i]}")
        print(f"Response: {response_text[:100]}...")
        print(f"Model A: {model_a_tokens}/{total_tokens} tokens ({100*model_a_tokens/max(total_tokens,1):.1f}%)")

    print("\nHuggingFace Output Sample:")
    for i in range(len(test_prompts)):
        response_ids = output_hf.batch["responses"][i]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        model_a_tokens = output_hf.batch["model_a_mask"][i].sum().item()
        total_tokens = (~(response_ids == tokenizer.pad_token_id)).sum().item()
        print(f"\nPrompt {i+1}: {test_prompts[i]}")
        print(f"Response: {response_text[:100]}...")
        print(f"Model A: {model_a_tokens}/{total_tokens} tokens ({100*model_a_tokens/max(total_tokens,1):.1f}%)")
    print("="*80)
