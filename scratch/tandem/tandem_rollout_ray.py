import torch
import ray
from concurrent.futures import ThreadPoolExecutor
from tensordict import TensorDict
from verl import DataProto
from verl.utils.torch_functional import get_response_mask


class TandemRolloutRay:
    def __init__(self, model_a, frozen_actor, tokenizer, config):
        self.model_a = model_a
        self.frozen_actor = frozen_actor
        self.tokenizer = tokenizer
        self.config = config

        self.device_a = next(model_a.parameters()).device

        print(f"[TandemRolloutRay] Hot model device: {self.device_a}")
        print(f"[TandemRolloutRay] Frozen model: Ray actor on separate GPU")

        self.prob_a = 0.5
        self.model_a.eval()

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
        attn_mask_prefix_a = attention_mask.to(self.device_a)

        prefix_length = prefix_a.shape[1]
        total_length = prefix_length + max_new_tokens

        out_seq_a = prefix_a.new_full((batch_size, total_length), pad_token_id)
        out_seq_a[:, :prefix_length] = prefix_a

        attn_mask_a = torch.zeros_like(out_seq_a)
        attn_mask_a[:, :prefix_length] = attn_mask_prefix_a

        model_a_mask = torch.zeros((batch_size, max_new_tokens), dtype=torch.bool, device=self.device_a)

        cur_pos_a = prefix_length
        generation_step = 0
        done_seqs_a = torch.zeros(batch_size, device=self.device_a, dtype=torch.bool)

        past_key_values_a = None
        batch_id = id(prompts)

        with ThreadPoolExecutor(max_workers=2) as executor:
            while generation_step < max_new_tokens:
                if generation_step == 0:
                    cur_input_ids_a = prefix_a
                    cur_attn_a = attn_mask_a[:, :cur_pos_a]
                else:
                    cur_input_ids_a = out_seq_a[:, cur_pos_a - 1 : cur_pos_a]
                    cur_attn_a = attn_mask_a[:, :cur_pos_a]

                future_a = executor.submit(self._forward_hot, cur_input_ids_a, cur_attn_a, past_key_values_a)
                future_b = executor.submit(self._forward_frozen, batch_id, cur_input_ids_a, cur_attn_a)

                outputs_a = future_a.result()
                outputs_b = future_b.result()

                past_key_values_a = outputs_a['past_key_values']

                logits_a = outputs_a['logits'][:, -1, :]
                logits_b = outputs_b['logits'][:, -1, :]

                candidate_token_a = self._sample(logits_a, temperature)
                candidate_token_b = self._sample(logits_b, temperature)

                is_turn_a = torch.rand(batch_size, device=self.device_a) < self.prob_a

                chosen_token = torch.where(is_turn_a, candidate_token_a, candidate_token_b)

                out_seq_a[:, cur_pos_a] = torch.where(done_seqs_a, pad_token_id, chosen_token)
                model_a_mask[:, generation_step] = is_turn_a & ~done_seqs_a
                attn_mask_a[:, cur_pos_a] = ~done_seqs_a
                done_seqs_a = done_seqs_a | (chosen_token == eos_token_id)

                cur_pos_a += 1
                generation_step += 1

                if done_seqs_a.all():
                    break

        ray.get(self.frozen_actor.clear_cache.remote(batch_id))

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

    def _forward_hot(self, input_ids, attention_mask, past_key_values):
        from verl.utils.device import get_device_name
        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            outputs = self.model_a(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        return {
            'logits': outputs.logits,
            'past_key_values': outputs.past_key_values
        }

    def _forward_frozen(self, batch_id, input_ids, attention_mask):
        result = ray.get(self.frozen_actor.forward.remote(
            batch_id,
            input_ids.cpu(),
            attention_mask.cpu()
        ))
        return {'logits': result['logits'].to(self.device_a)}

    def _sample(self, logits, temperature):
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return sampled.to(logits.device)
