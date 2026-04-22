# Copyright (c) 2025 BAAI. All rights reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def has_platform_helpers():
    try:
        from vllm_fl.platform import (  # noqa: F401
            get_qwen3_4b_nvidia_batched_token_candidates,
            set_scheduler_max_num_batched_tokens,
        )

        return True
    except Exception:
        return False


def has_worker_helpers():
    try:
        from vllm_fl.worker.worker import WorkerFL  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not has_platform_helpers(), reason="vllm_fl.platform helpers not available"
)
class TestQwen3SchedulerDefaults:
    def test_qwen3_4b_nvidia_candidates_use_default_ladder(self):
        from vllm_fl.platform import get_qwen3_4b_nvidia_batched_token_candidates

        platform_cls = SimpleNamespace(vendor_name="nvidia", device_type="cuda")
        scheduler_config = SimpleNamespace(
            max_num_batched_tokens=8192,
            _fl_user_provided_max_num_batched_tokens=False,
            _fl_upstream_default_max_num_batched_tokens=8192,
        )
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                model="Qwen/Qwen3-4B",
                hf_text_config=SimpleNamespace(model_type="qwen3"),
            ),
            cache_config=SimpleNamespace(cache_dtype="fp8"),
            scheduler_config=scheduler_config,
        )

        assert get_qwen3_4b_nvidia_batched_token_candidates(platform_cls, vllm_config) == [
            16384,
            12288,
            8192,
        ]

    def test_qwen3_4b_candidates_skip_user_override(self):
        from vllm_fl.platform import get_qwen3_4b_nvidia_batched_token_candidates

        platform_cls = SimpleNamespace(vendor_name="nvidia", device_type="cuda")
        scheduler_config = SimpleNamespace(
            max_num_batched_tokens=8192,
            _fl_user_provided_max_num_batched_tokens=True,
            _fl_upstream_default_max_num_batched_tokens=8192,
        )
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                model="Qwen/Qwen3-4B",
                hf_text_config=SimpleNamespace(model_type="qwen3"),
            ),
            cache_config=SimpleNamespace(cache_dtype="fp8"),
            scheduler_config=scheduler_config,
        )

        assert get_qwen3_4b_nvidia_batched_token_candidates(platform_cls, vllm_config) is None

    def test_set_scheduler_max_num_batched_tokens_updates_related_limits(self):
        from vllm_fl.platform import set_scheduler_max_num_batched_tokens

        scheduler_config = SimpleNamespace(
            max_num_batched_tokens=8192,
            max_num_encoder_input_tokens=8192,
            encoder_cache_size=8192,
        )

        set_scheduler_max_num_batched_tokens(scheduler_config, 12288)

        assert scheduler_config.max_num_batched_tokens == 12288
        assert scheduler_config.max_num_encoder_input_tokens == 12288
        assert scheduler_config.encoder_cache_size == 12288


@pytest.mark.skipif(
    not has_worker_helpers(), reason="vllm_fl.worker.worker not available"
)
class TestQwen3SchedulerFallback:
    def test_worker_oom_fallback_uses_next_candidate(self):
        from vllm_fl.worker.worker import WorkerFL

        scheduler_config = SimpleNamespace(
            max_num_batched_tokens=16384,
            max_num_encoder_input_tokens=16384,
            encoder_cache_size=16384,
            _fl_qwen3_4b_batched_token_candidates=[16384, 12288, 8192],
        )
        worker = SimpleNamespace(
            scheduler_config=scheduler_config,
            _rebuild_model_runner_for_scheduler_retry=MagicMock(),
        )

        applied = WorkerFL._maybe_apply_qwen3_4b_oom_fallback(
            worker,
            "initial profiling",
            RuntimeError("CUDA out of memory"),
        )

        assert applied is True
        assert scheduler_config.max_num_batched_tokens == 12288
        assert scheduler_config.max_num_encoder_input_tokens == 12288
        assert scheduler_config.encoder_cache_size == 12288
        worker._rebuild_model_runner_for_scheduler_retry.assert_called_once_with()

    def test_worker_oom_fallback_stops_at_last_candidate(self):
        from vllm_fl.worker.worker import WorkerFL

        scheduler_config = SimpleNamespace(
            max_num_batched_tokens=8192,
            max_num_encoder_input_tokens=8192,
            encoder_cache_size=8192,
            _fl_qwen3_4b_batched_token_candidates=[16384, 12288, 8192],
        )
        worker = SimpleNamespace(
            scheduler_config=scheduler_config,
            _rebuild_model_runner_for_scheduler_retry=MagicMock(),
        )

        applied = WorkerFL._maybe_apply_qwen3_4b_oom_fallback(
            worker,
            "initial profiling",
            RuntimeError("CUDA out of memory"),
        )

        assert applied is False
        worker._rebuild_model_runner_for_scheduler_retry.assert_not_called()
