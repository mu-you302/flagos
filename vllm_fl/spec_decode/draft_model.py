# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations

import copy
import os
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn

from vllm.compilation.backends import set_model_tag
from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

logger = init_logger(__name__)

PADDING_SLOT_ID = -1


class DraftModelProposer:
    """A minimal draft-model proposer for text-only Qwen-style models.

    vLLM 0.13 has the scheduler and rejection sampler pieces for speculative
    decoding, but intentionally rejects plain draft-model speculation. This
    proposer supplies the missing draft-token generation path for the Qwen3-4B
    benchmark use case.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner: Any,
    ) -> None:
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        assert self.speculative_config.draft_model_config is not None

        self.draft_model_config = self.speculative_config.draft_model_config
        self.device = device
        self.runner = runner
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens
        self.block_size = vllm_config.cache_config.block_size
        self.max_model_len = self.draft_model_config.max_model_len

        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.arange = torch.arange(max_num_reqs + 1, device=device, dtype=torch.int32)
        self.input_ids = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.positions = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)

        self.model: nn.Module | None = None
        self.attn_layer_names: list[str] = []
        self.attn_metadata_builder: AttentionMetadataBuilder | None = None
        self._draft_probs_by_req: dict[str, list[torch.Tensor]] = {}
        self.min_acceptance_rate = float(
            os.environ.get("VLLM_FL_DRAFT_MIN_ACCEPTANCE_RATE", "0.2")
        )
        self.low_acceptance_patience = int(
            os.environ.get("VLLM_FL_DRAFT_LOW_ACCEPTANCE_PATIENCE", "2")
        )
        self.low_acceptance_cooldown = int(
            os.environ.get("VLLM_FL_DRAFT_LOW_ACCEPTANCE_COOLDOWN", "64")
        )
        self._low_acceptance_steps = 0
        self._cooldown_remaining = 0

    @contextmanager
    def _draft_init_config_scope(self):
        target_model_config = self.vllm_config.model_config
        target_quant_config = self.vllm_config.quant_config
        target_cache_config = self.vllm_config.cache_config

        draft_cache_config = copy.deepcopy(target_cache_config)
        draft_cache_config.cache_dtype = "auto"
        draft_cache_config.calculate_kv_scales = False
        draft_quant_config = type(self.vllm_config).get_quantization_config(
            self.draft_model_config, self.vllm_config.load_config
        )

        self.vllm_config.model_config = self.draft_model_config
        self.vllm_config.quant_config = draft_quant_config
        self.vllm_config.cache_config = draft_cache_config
        try:
            yield
        finally:
            self.vllm_config.model_config = target_model_config
            self.vllm_config.quant_config = target_quant_config
            self.vllm_config.cache_config = target_cache_config

    def load_model(self, target_model: nn.Module) -> None:
        del target_model  # The plain draft model has independent weights.

        existing_attn_layers = set(
            get_layers_from_vllm_config(
                self.vllm_config, AttentionLayerBase
            ).keys()
        )

        model_loader = get_model_loader(self.vllm_config.load_config)
        device_config = self.vllm_config.device_config
        load_config = self.vllm_config.load_config
        load_device = (
            device_config.device if load_config.device is None else load_config.device
        )
        target_device = torch.device(load_device)

        with set_model_tag("draft_model"):
            with set_default_torch_dtype(self.draft_model_config.dtype):
                with self._draft_init_config_scope():
                    with target_device:
                        self.model = initialize_model(
                            vllm_config=self.vllm_config,
                            model_config=self.draft_model_config,
                            prefix="draft_model",
                        )

                logger.debug("Loading draft model weights on %s ...", load_device)
                model_loader.load_weights(self.model, self.draft_model_config)
                process_weights_after_loading(
                    self.model, self.draft_model_config, target_device
                )

        self.model.eval()
        for module in self.model.modules():
            if hasattr(module, "do_not_compile"):
                module.do_not_compile = True

        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config, AttentionLayerBase
        )
        self.attn_layer_names = [
            name for name in all_attn_layers if name not in existing_attn_layers
        ]
        if not self.attn_layer_names:
            raise RuntimeError("Draft model did not register any attention layers.")

        logger.info(
            "Loaded draft model %s with %d attention layers.",
            self.draft_model_config.model,
            len(self.attn_layer_names),
        )

    def take_draft_probs(
        self,
        expected_num_tokens: int,
        req_ids: list[str] | None = None,
        num_draft_tokens: list[int] | None = None,
    ) -> torch.Tensor | None:
        if not self._draft_probs_by_req:
            return None
        if req_ids is None or num_draft_tokens is None:
            rows = [
                row
                for req_rows in self._draft_probs_by_req.values()
                for row in req_rows
            ]
            self._draft_probs_by_req.clear()
        else:
            rows = []
            for req_id, num_tokens in zip(req_ids, num_draft_tokens):
                req_rows = self._draft_probs_by_req.pop(req_id, [])
                if len(req_rows) < num_tokens:
                    logger.warning(
                        "Discarding draft probabilities for request %s: "
                        "expected %d rows, got %d.",
                        req_id,
                        num_tokens,
                        len(req_rows),
                    )
                    self._draft_probs_by_req.clear()
                    return None
                rows.extend(req_rows[:num_tokens])

        if len(rows) != expected_num_tokens:
            logger.warning(
                "Discarding draft probabilities: expected %d rows, got %d.",
                expected_num_tokens,
                len(rows),
            )
            return None
        return torch.stack(rows, dim=0).contiguous() if rows else None

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        req_ids: list[str],
        requests: dict[str, CachedRequestState],
        input_batch: InputBatch,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        spec_decode_unsupported_reqs: set,
        previous_spec_decode_metadata: Any | None = None,
    ) -> list[list[int]]:
        if self.model is None:
            raise RuntimeError("Draft model has not been loaded.")

        num_reqs = len(req_ids)
        if num_reqs == 0 or not get_pp_group().is_last_rank:
            self._draft_probs_by_req.clear()
            return [[] for _ in req_ids]

        if previous_spec_decode_metadata is not None:
            self._update_adaptive_state(
                sampled_token_ids,
                previous_spec_decode_metadata.num_draft_tokens,
            )
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self._draft_probs_by_req.clear()
            return [[] for _ in req_ids]

        self._draft_probs_by_req.clear()
        base_seq_lens = self._get_base_seq_lens(input_batch, num_reqs)
        next_token_ids = self._prepare_next_token_ids(
            sampled_token_ids,
            req_ids,
            requests,
            base_seq_lens,
        )
        valid_mask = self._initial_valid_mask(
            sampled_token_ids,
            req_ids,
            spec_decode_unsupported_reqs,
            base_seq_lens,
        )

        draft_token_ids: list[list[int]] = [[] for _ in req_ids]
        draft_prob_rows: list[list[torch.Tensor]] = [[] for _ in req_ids]

        for draft_index in range(self.num_speculative_tokens):
            if not bool(valid_mask.any().item()):
                break

            positions = base_seq_lens + draft_index - 1
            step_metadata = self._build_step_metadata(
                common_attn_metadata=common_attn_metadata,
                positions=positions,
                valid_mask=valid_mask,
            )
            logits = self._forward_step(
                input_ids=next_token_ids,
                positions=positions,
                common_attn_metadata=step_metadata,
                draft_index=draft_index,
            )

            if sampling_metadata.all_greedy:
                next_token_ids = logits.argmax(dim=-1).to(torch.int32)
            else:
                probs = logits.softmax(dim=-1, dtype=torch.float32)
                sampled_token_ids_tensor = torch.multinomial(
                    probs, num_samples=1
                ).squeeze(1)
                next_token_ids = sampled_token_ids_tensor.to(torch.int32)
                if (
                    not sampling_metadata.all_random
                    and sampling_metadata.temperature is not None
                ):
                    greedy_mask = sampling_metadata.temperature[:num_reqs] == 0
                    greedy_token_ids = logits.argmax(dim=-1).to(torch.int32)
                    next_token_ids = torch.where(
                        greedy_mask,
                        greedy_token_ids,
                        next_token_ids,
                    )
                active_indices_cpu = (
                    valid_mask.nonzero(as_tuple=False).flatten().cpu()
                )
                for req_index in active_indices_cpu.tolist():
                    draft_prob_rows[req_index].append(
                        probs[req_index].to(dtype=logits.dtype).contiguous()
                    )

            next_token_ids_cpu = next_token_ids.detach().cpu().tolist()
            active_indices_cpu = valid_mask.nonzero(as_tuple=False).flatten().cpu()
            for req_index in active_indices_cpu.tolist():
                draft_token_ids[req_index].append(int(next_token_ids_cpu[req_index]))

            valid_mask = valid_mask & (positions + 1 < self.max_model_len)

        self._draft_probs_by_req = {
            req_id: rows for req_id, rows in zip(req_ids, draft_prob_rows) if rows
        }
        return draft_token_ids

    def _update_adaptive_state(
        self,
        sampled_token_ids: list[list[int]],
        num_draft_tokens: list[int],
    ) -> None:
        if (
            self.min_acceptance_rate <= 0
            or self.low_acceptance_patience <= 0
            or self.low_acceptance_cooldown <= 0
        ):
            return

        total_draft = sum(num_draft_tokens)
        if total_draft == 0:
            return

        accepted_draft = 0
        for output_tokens, num_draft in zip(sampled_token_ids, num_draft_tokens):
            accepted_draft += min(max(len(output_tokens) - 1, 0), num_draft)

        acceptance_rate = accepted_draft / total_draft
        if acceptance_rate < self.min_acceptance_rate:
            self._low_acceptance_steps += 1
            if self._low_acceptance_steps >= self.low_acceptance_patience:
                self._cooldown_remaining = self.low_acceptance_cooldown
                self._low_acceptance_steps = 0
                logger.info(
                    "Draft model speculative decoding acceptance rate %.2f is "
                    "below %.2f for %d checked steps; pausing draft proposals "
                    "for %d decode steps.",
                    acceptance_rate,
                    self.min_acceptance_rate,
                    self.low_acceptance_patience,
                    self.low_acceptance_cooldown,
                )
        else:
            self._low_acceptance_steps = 0

    def _get_base_seq_lens(
        self,
        input_batch: InputBatch,
        num_reqs: int,
    ) -> torch.Tensor:
        seq_lens = [
            int(input_batch.num_tokens_no_spec[req_index])
            for req_index in range(num_reqs)
        ]
        return torch.tensor(seq_lens, dtype=torch.int64, device=self.device)

    def _prepare_next_token_ids(
        self,
        sampled_token_ids: list[list[int]],
        req_ids: list[str],
        requests: dict[str, CachedRequestState],
        base_seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        next_token_ids: list[int] = []
        base_seq_lens_cpu = base_seq_lens.detach().cpu()
        for req_index, sampled_ids in enumerate(sampled_token_ids):
            if sampled_ids:
                next_token_ids.append(sampled_ids[-1])
            else:
                req = requests[req_ids[req_index]]
                token_index = max(int(base_seq_lens_cpu[req_index].item()) - 1, 0)
                next_token_ids.append(req.get_token_id(token_index))

        return torch.tensor(next_token_ids, dtype=torch.int32, device=self.device)

    def _initial_valid_mask(
        self,
        sampled_token_ids: list[list[int]],
        req_ids: list[str],
        spec_decode_unsupported_reqs: set,
        base_seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        valid = []
        base_seq_lens_cpu = base_seq_lens.detach().cpu()
        for req_index, sampled_ids in enumerate(sampled_token_ids):
            req_id = req_ids[req_index]
            seq_len = int(base_seq_lens_cpu[req_index].item())
            valid.append(
                bool(sampled_ids)
                and req_id not in spec_decode_unsupported_reqs
                and 0 < seq_len < self.max_model_len
            )
        return torch.tensor(valid, dtype=torch.bool, device=self.device)

    def _build_step_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        positions: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> CommonAttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        clamped_positions = torch.where(
            valid_mask, positions, torch.zeros_like(positions)
        )
        seq_lens = clamped_positions.to(torch.int32) + 1
        seq_lens = torch.where(valid_mask, seq_lens, torch.ones_like(seq_lens))

        block_numbers = clamped_positions // self.block_size
        block_ids = common_attn_metadata.block_table_tensor.gather(
            dim=1, index=block_numbers.view(-1, 1)
        ).view(-1)
        slot_mapping = block_ids * self.block_size + clamped_positions % self.block_size
        slot_mapping = slot_mapping.to(torch.int64)
        slot_mapping.masked_fill_(~valid_mask, PADDING_SLOT_ID)

        query_start_loc = self.arange[: num_reqs + 1]
        query_start_loc_cpu = torch.arange(num_reqs + 1, dtype=torch.int32)
        seq_lens_cpu = seq_lens.detach().cpu()

        return CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            num_reqs=num_reqs,
            num_actual_tokens=num_reqs,
            max_query_len=1,
            max_seq_len=int(seq_lens_cpu.max().item()),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=slot_mapping,
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
            dcp_local_seq_lens_cpu=common_attn_metadata.dcp_local_seq_lens_cpu,
            _seq_lens_cpu=seq_lens_cpu,
            _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
        )

    def _forward_step(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
    ) -> torch.Tensor:
        assert self.model is not None
        attn_metadata_builder = self._get_attention_metadata_builder()
        attn_metadata = attn_metadata_builder.build_for_drafting(
            common_attn_metadata=common_attn_metadata,
            draft_index=draft_index,
        )
        per_layer_attn_metadata = {
            layer_name: attn_metadata for layer_name in self.attn_layer_names
        }

        num_tokens = common_attn_metadata.num_actual_tokens
        self.input_ids[:num_tokens] = input_ids[:num_tokens]
        self.positions[:num_tokens] = positions[:num_tokens]

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        ):
            hidden_states = self.model(
                input_ids=self.input_ids[:num_tokens],
                positions=self.positions[:num_tokens],
                intermediate_tensors=None,
                inputs_embeds=None,
            )
        return self.model.compute_logits(hidden_states)

    def _get_attention_metadata_builder(self) -> AttentionMetadataBuilder:
        if self.attn_metadata_builder is not None:
            return self.attn_metadata_builder

        chosen_layer = self.attn_layer_names[0]
        for kv_cache_group in self.runner.attn_groups:
            for attn_group in kv_cache_group:
                if chosen_layer in attn_group.layer_names:
                    self.attn_metadata_builder = attn_group.get_metadata_builder()
                    return self.attn_metadata_builder

        raise RuntimeError(
            f"Failed to find attention metadata builder for draft layer {chosen_layer}."
        )
