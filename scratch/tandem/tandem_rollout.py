import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from tensordict import TensorDict

from verl import DataProto
from verl.utils.torch_functional import get_response_mask


class TandemRollout:
    def __init__(self, model_a, model_b, tokenizer, config):
        self.model_a = model_a
        self.model_b = model_b
        self.tokenizer = tokenizer
        self.config = config

        # Device detection - models should already be on GPU (non-FSDP)
        self.device_a = next(model_a.parameters()).device
        self.device_b = next(model_b.parameters()).device

        print(f"[TandemRollout] Hot model (model_a) device: {self.device_a}")
        print(f"[TandemRollout] Frozen model (model_b) device: {self.device_b}")
        if self.device_a == self.device_b:
            print(f"[TandemRollout] WARNING: Both models on same GPU - NO parallel execution!")
        else:
            print(f"[TandemRollout] Models on different GPUs - TRUE parallel execution enabled!")

        self.prob_a = 0.5
        # Set models to eval mode for generation
        self.model_a.eval()
        self.model_b.eval()

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

        input_ids = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        batch_size = input_ids.size(0)
        prompt_length = input_ids.size(1)
        max_new_tokens = response_length

        prefix_a = input_ids.to(self.device_a)
        prefix_b = input_ids.to(self.device_b)
        attn_mask_prefix_a = attention_mask.to(self.device_a)
        attn_mask_prefix_b = attention_mask.to(self.device_b)

        prefix_length = prefix_a.shape[1]
        total_length = prefix_length + max_new_tokens

        out_seq_a = prefix_a.new_full((batch_size, total_length), pad_token_id)
        out_seq_a[:, :prefix_length] = prefix_a
        out_seq_b = prefix_b.new_full((batch_size, total_length), pad_token_id)
        out_seq_b[:, :prefix_length] = prefix_b

        attn_mask_a = torch.zeros_like(out_seq_a)
        attn_mask_a[:, :prefix_length] = attn_mask_prefix_a
        attn_mask_b = torch.zeros_like(out_seq_b)
        attn_mask_b[:, :prefix_length] = attn_mask_prefix_b

        model_a_mask = torch.zeros((batch_size, max_new_tokens), dtype=torch.bool, device=self.device_a)

        cur_pos_a = prefix_length
        cur_pos_b = prefix_length
        generation_step = 0
        done_seqs_a = torch.zeros(batch_size, device=self.device_a, dtype=torch.bool)
        done_seqs_b = torch.zeros(batch_size, device=self.device_b, dtype=torch.bool)

        with ThreadPoolExecutor(max_workers=16) as executor:
            while generation_step < max_new_tokens:
                if generation_step == 0:
                    cur_input_ids_a = prefix_a
                    cur_input_ids_b = prefix_b
                    past_key_values_a = None
                    past_key_values_b = None
                else:
                    cur_input_ids_a = out_seq_a[:, cur_pos_a - 1 : cur_pos_a]
                    cur_input_ids_b = out_seq_b[:, cur_pos_b - 1 : cur_pos_b]
                    past_key_values_a = caches_a
                    past_key_values_b = caches_b

                future_a = executor.submit(
                    self._model_forward,
                    self.model_a,
                    cur_input_ids_a,
                    attn_mask_a[:, :cur_pos_a],
                    past_key_values_a,
                )
                future_b = executor.submit(
                    self._model_forward,
                    self.model_b,
                    cur_input_ids_b,
                    attn_mask_b[:, :cur_pos_b],
                    past_key_values_b,
                )

                outputs_a = future_a.result()
                outputs_b = future_b.result()

                caches_a = outputs_a.past_key_values
                caches_b = outputs_b.past_key_values

                logits_a = outputs_a.logits[:, -1, :]
                logits_b = outputs_b.logits[:, -1, :]

                candidate_token_a = self._sample(logits_a, temperature)
                candidate_token_b = self._sample(logits_b, temperature).to(self.device_a)

                is_turn_a = torch.rand(batch_size, device=self.device_a) < self.prob_a

                # Ensure all tensors are on same device for torch.where
                chosen_token = torch.where(
                    is_turn_a.to(candidate_token_a.device),
                    candidate_token_a,
                    candidate_token_b.to(candidate_token_a.device)
                )

                chosen_token_a = chosen_token.to(out_seq_a.device)
                chosen_token_b = chosen_token.to(out_seq_b.device)

                out_seq_a[:, cur_pos_a] = torch.where(done_seqs_a, pad_token_id, chosen_token_a)
                out_seq_b[:, cur_pos_b] = torch.where(done_seqs_b, pad_token_id, chosen_token_b)

                model_a_mask[:, generation_step] = is_turn_a & ~done_seqs_a

                attn_mask_a[:, cur_pos_a] = ~done_seqs_a
                attn_mask_b[:, cur_pos_b] = ~done_seqs_b

                done_seqs_a = done_seqs_a | (chosen_token_a == eos_token_id)
                done_seqs_b = done_seqs_b | (chosen_token_b == eos_token_id)

                cur_pos_a += 1
                cur_pos_b += 1
                generation_step += 1

                if done_seqs_a.all():
                    break

        seq = out_seq_a
        prompt = seq[:, :prompt_length]
        response = seq[:, prompt_length:prompt_length + max_new_tokens]

        response_length_actual = response.size(1)
        delta_position_id = torch.arange(1, response_length_actual + 1, device=self.device_a)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        response_position_ids = position_ids.to(self.device_a)[:, -1:] + delta_position_id
        position_ids_full = torch.cat([position_ids.to(self.device_a), response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask_full = torch.cat((attention_mask.to(self.device_a), response_attention_mask), dim=-1)

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

    def _model_forward(self, model, input_ids, attention_mask, past_key_values):
        # Use bfloat16 autocast for efficiency
        from verl.utils.device import get_device_name

        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            return model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

    def _sample(self, logits, temperature):
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        # Ensure sampled tokens stay on the same device as logits
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return sampled.to(logits.device)
