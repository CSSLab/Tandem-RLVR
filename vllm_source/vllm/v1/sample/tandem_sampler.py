# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import torch.nn as nn

from vllm.config import TandemConfig
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler


class TandemSampler(nn.Module):

    def __init__(self, tandem_config: TandemConfig):
        super().__init__()
        self.strategy = tandem_config.selection_strategy
        self.prob_primary = tandem_config.prob_primary
        self.chunk_size = max(1, getattr(tandem_config, 'chunk_size', 1))
        self.max_gap_tokens = max(1, getattr(tandem_config, 'max_gap_tokens', 32))
        self.sampler = Sampler()
        self._step = 0
        self._word_state: dict = {}
        self.boundary_token_ids: frozenset[int] = (
            frozenset(tandem_config.boundary_token_ids)
            if tandem_config.boundary_token_ids else frozenset()
        )

    def forward(
        self,
        primary_logits: torch.Tensor,
        frozen_logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[SamplerOutput, torch.Tensor]:
        batch_size = primary_logits.shape[0]

        primary_output = self.sampler(primary_logits.clone(), sampling_metadata)
        frozen_output = self.sampler(frozen_logits.clone(), sampling_metadata)

        primary_tokens = primary_output.sampled_token_ids.squeeze(-1)
        frozen_tokens = frozen_output.sampled_token_ids.squeeze(-1)

        use_primary = self._select(batch_size, primary_logits.device,
                                    sampling_metadata)

        chosen_tokens = torch.where(use_primary, primary_tokens, frozen_tokens)
        model_mask = use_primary.to(torch.int32)

        self._step += 1

        sampler_output = SamplerOutput(
            sampled_token_ids=chosen_tokens.unsqueeze(-1),
            logprobs_tensors=primary_output.logprobs_tensors,
        )
        return sampler_output, model_mask

    def _select(self, batch_size: int, device: torch.device,
                 sampling_metadata: SamplingMetadata) -> torch.Tensor:
        if self.strategy == "bernoulli":
            return torch.rand(batch_size, device=device) < self.prob_primary
        elif self.strategy == "chunk":
            return self._chunk_select(batch_size, device)
        elif self.strategy == "alternating":
            return torch.full((batch_size,), self._step % 2 == 0,
                              dtype=torch.bool, device=device)
        elif self.strategy == "sentence":
            return self._sentence_select(batch_size, device,
                                         sampling_metadata)
        elif self.strategy == "word":
            return self._word_select(batch_size, device,
                                     sampling_metadata)
        raise ValueError(f"Unknown strategy: {self.strategy}")

    def _sentence_select(self, batch_size: int, device: torch.device,
                          sampling_metadata: SamplingMetadata) -> torch.Tensor:
        use_primary = torch.empty(batch_size, dtype=torch.bool, device=device)
        boundary_ids = self.boundary_token_ids
        for i, token_ids in enumerate(sampling_metadata.output_token_ids):
            n_boundaries = sum(1 for tid in token_ids if tid in boundary_ids)
            use_primary[i] = (n_boundaries % 2 == 0)
        return use_primary

    def _word_select(self, batch_size: int, device: torch.device,
                      sampling_metadata: SamplingMetadata) -> torch.Tensor:
        # [MODIFIED 2026-05-05 per-boundary Bernoulli(p) handoffs (was strict toggle); matches West et al. 2026]
        use_primary = torch.empty(batch_size, dtype=torch.bool, device=device)
        boundary_ids = self.boundary_token_ids
        max_gap = self.max_gap_tokens
        p = self.prob_primary
        for i, token_ids in enumerate(sampling_metadata.output_token_ids):
            key = id(token_ids)
            cur_len = len(token_ids)
            cached = self._word_state.get(key)
            if cached is None or cached[2] > cur_len:
                is_senior = self._draw_active(sampling_metadata, i, p)
                since_last_switch = 0
                start = 0
            else:
                is_senior, since_last_switch, start = cached
            for tid in token_ids[start:]:
                since_last_switch += 1
                if tid in boundary_ids or since_last_switch >= max_gap:
                    is_senior = self._draw_active(sampling_metadata, i, p)
                    since_last_switch = 0
            self._word_state[key] = (is_senior, since_last_switch, cur_len)
            use_primary[i] = is_senior
        return use_primary

    def _draw_active(self, sampling_metadata: SamplingMetadata, i: int,
                      p: float) -> bool:
        gen = sampling_metadata.generators.get(i)
        if gen is not None:
            draw = torch.rand(1, generator=gen, device=gen.device).item()
        else:
            draw = float(torch.rand(1).item())
        return draw < p

    def _chunk_select(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # [MODIFIED 2026-04-16 use explicit chunk_size instead of deriving from prob_primary]
        # chunk_size = max(1, int(1.0 / (1.0 - self.prob_primary + 1e-9)))
        # cycle_len = chunk_size + max(1, int(1.0 / (self.prob_primary + 1e-9)))
        pos = self._step % (2 * self.chunk_size)
        is_primary = pos < self.chunk_size
        return torch.full((batch_size,), is_primary,
                          dtype=torch.bool, device=device)

    def reset(self):
        self._step = 0
        self._word_state = {}
