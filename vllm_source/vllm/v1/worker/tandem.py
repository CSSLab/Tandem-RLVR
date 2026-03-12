# SPDX-License-Identifier: Apache-2.0

import copy
import time
from typing import Any, Optional

import torch
from torch import nn

from vllm.attention import Attention
from vllm.config import TandemConfig, VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger

logger = init_logger(__name__)

FROZEN_PREFIX = "tandem_frozen."


def is_frozen_layer(layer_name: str) -> bool:
    return layer_name.startswith(FROZEN_PREFIX)


class TandemModelManager:

    def __init__(self, tandem_config: TandemConfig, primary_vllm_config: VllmConfig):
        self.tandem_config = tandem_config
        self.primary_vllm_config = primary_vllm_config
        self.frozen_model: Optional[nn.Module] = None
        self.frozen_device: Optional[torch.device] = None
        self.frozen_kv_caches: list[torch.Tensor] = []

    def _build_frozen_vllm_config(self) -> VllmConfig:
        tc = self.tandem_config
        primary = self.primary_vllm_config

        frozen_config = copy.deepcopy(primary)

        if tc.frozen_model != primary.model_config.model:
            frozen_config.model_config.model = tc.frozen_model
            if tc.frozen_model_revision:
                frozen_config.model_config.revision = tc.frozen_model_revision
        if tc.frozen_quantization:
            frozen_config.model_config.quantization = tc.frozen_quantization
        if tc.frozen_max_model_len:
            frozen_config.model_config.max_model_len = tc.frozen_max_model_len

        return frozen_config

    def load_frozen_model(self) -> None:
        tc = self.tandem_config
        if not tc.enabled:
            return

        if tc.frozen_gpu_devices:
            device_id = tc.frozen_gpu_devices[0]
        else:
            primary_device = self.primary_vllm_config.device_config.device
            if hasattr(primary_device, 'index') and primary_device.index is not None:
                device_id = primary_device.index + 1
            else:
                device_id = 1
        self.frozen_device = torch.device(f"cuda:{device_id}")

        logger.info("Loading frozen tandem model %s on %s...",
                     tc.frozen_model, self.frozen_device)

        frozen_vllm_config = self._build_frozen_vllm_config()

        from vllm.model_executor.model_loader.loader import (
            DefaultModelLoader, _initialize_model)
        from vllm.model_executor.model_loader.utils import set_default_torch_dtype

        loader = DefaultModelLoader(frozen_vllm_config.load_config)

        time_before = time.perf_counter()
        with set_default_torch_dtype(frozen_vllm_config.model_config.dtype):
            with torch.device(self.frozen_device):
                self.frozen_model = _initialize_model(
                    vllm_config=frozen_vllm_config,
                    prefix=FROZEN_PREFIX,
                )

            self.frozen_model.load_weights(
                loader.get_all_weights(
                    frozen_vllm_config.model_config, self.frozen_model))

        self.frozen_model.eval()
        for param in self.frozen_model.parameters():
            param.requires_grad_(False)

        frozen_sfc = frozen_vllm_config.compilation_config.static_forward_context
        primary_sfc = self.primary_vllm_config.compilation_config.static_forward_context
        frozen_attn_count = 0
        for name, layer in frozen_sfc.items():
            if is_frozen_layer(name) and name not in primary_sfc:
                primary_sfc[name] = layer
                frozen_attn_count += 1

        time_after = time.perf_counter()
        logger.info("Frozen tandem model loaded on %s in %.2fs (%d attn layers merged)",
                     self.frozen_device, time_after - time_before, frozen_attn_count)

    def initialize_frozen_kv_cache(
        self,
        primary_kv_cache_config: Any,
        attn_backend: Any,
    ) -> None:
        if self.frozen_model is None:
            return

        from vllm.v1.kv_cache_interface import AttentionSpec
        from vllm.v1.utils import bind_kv_cache

        sfc = self.primary_vllm_config.compilation_config.static_forward_context
        frozen_layers = {
            name: layer for name, layer in sfc.items()
            if is_frozen_layer(name) and isinstance(layer, Attention)
        }

        if not frozen_layers:
            logger.warning("No frozen attention layers found in static_forward_context")
            return

        num_blocks = primary_kv_cache_config.num_blocks
        block_size = self.primary_vllm_config.cache_config.block_size
        kv_cache_dtype = self.primary_vllm_config.cache_config.cache_dtype
        if kv_cache_dtype == "auto":
            kv_cache_dtype = self.primary_vllm_config.model_config.dtype

        frozen_kv_caches: dict[str, torch.Tensor] = {}
        for layer_name, attn_module in frozen_layers.items():
            kv_cache_shape = attn_backend.get_kv_cache_shape(
                num_blocks, block_size,
                attn_module.num_kv_heads, attn_module.head_size)
            frozen_kv_caches[layer_name] = torch.zeros(
                kv_cache_shape, dtype=kv_cache_dtype, device=self.frozen_device)

        bind_kv_cache(frozen_kv_caches, sfc, self.frozen_kv_caches)

        logger.info("Frozen tandem KV cache initialized: %d layers, %d blocks on %s",
                     len(frozen_kv_caches), num_blocks, self.frozen_device)

    def frozen_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: Any,
        num_scheduled_tokens: int,
        logits_indices: torch.Tensor,
    ) -> torch.Tensor:
        frozen_input_ids = input_ids.to(self.frozen_device)
        frozen_positions = positions.to(self.frozen_device)
        frozen_logits_indices = logits_indices.to(self.frozen_device)

        frozen_attn_metadata = self._adapt_attn_metadata(attn_metadata)

        with set_forward_context(frozen_attn_metadata, self.primary_vllm_config):
            hidden_states = self.frozen_model(
                input_ids=frozen_input_ids,
                positions=frozen_positions,
                intermediate_tensors=None,
                inputs_embeds=None,
            )

        hidden_states = hidden_states[:num_scheduled_tokens]
        sample_hidden_states = hidden_states[frozen_logits_indices]
        frozen_logits = self.frozen_model.compute_logits(
            sample_hidden_states, None)

        return frozen_logits

    def _adapt_attn_metadata(self, primary_metadata: Any) -> Any:
        adapted = copy.copy(primary_metadata)
        if hasattr(adapted, 'slot_mapping') and adapted.slot_mapping is not None:
            adapted.slot_mapping = adapted.slot_mapping.to(self.frozen_device)
        if hasattr(adapted, 'block_table') and adapted.block_table is not None:
            adapted.block_table = adapted.block_table.to(self.frozen_device)
        if hasattr(adapted, 'query_start_loc') and adapted.query_start_loc is not None:
            adapted.query_start_loc = adapted.query_start_loc.to(self.frozen_device)
        if hasattr(adapted, 'seq_lens') and adapted.seq_lens is not None:
            adapted.seq_lens = adapted.seq_lens.to(self.frozen_device)
        return adapted

    def get_frozen_model(self) -> nn.Module:
        if self.frozen_model is None:
            raise RuntimeError(
                "Frozen model not loaded. Call load_frozen_model() first.")
        return self.frozen_model

    def get_frozen_device(self) -> torch.device:
        if self.frozen_device is None:
            raise RuntimeError(
                "Frozen device not set. Call load_frozen_model() first.")
        return self.frozen_device
