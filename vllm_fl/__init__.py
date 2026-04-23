# Copyright (c) 2025 BAAI. All rights reserved.

import os
import logging

from . import version as version  # PyTorch-style: vllm_fl.version.git_version


logger = logging.getLogger(__name__)


_QWEN3_4B_BENCHMARK_DEFAULTS = {
    (1024, 1024): {
        "max_model_len": 2048,
        "max_num_batched_tokens": 16384,
        "max_num_seqs": 512,
    },
    (4096, 1024): {
        "max_model_len": 6144,
        "max_num_batched_tokens": 12288,
        "max_num_seqs": 256,
    },
    (6144, 1024): {
        "max_model_len": 8192,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 128,
    },
}


def _normalize_model_name(model) -> str:
    if not isinstance(model, str):
        return ""
    return model.rstrip("/").split("/")[-1].lower()


def _get_qwen3_4b_benchmark_defaults(args) -> dict[str, int] | None:
    input_len = getattr(args, "input_len", None)
    output_len = getattr(args, "output_len", None)
    if not isinstance(input_len, int) or not isinstance(output_len, int):
        return None

    model_name = _normalize_model_name(getattr(args, "model", None))
    if "qwen3-4b" not in model_name:
        return None

    return _QWEN3_4B_BENCHMARK_DEFAULTS.get((input_len, output_len))


def _apply_qwen3_4b_benchmark_defaults(engine_args, args) -> dict[str, int]:
    defaults = _get_qwen3_4b_benchmark_defaults(args)
    if not defaults:
        return {}

    applied: dict[str, int] = {}
    for field_name, value in defaults.items():
        if getattr(engine_args, field_name, None) is None:
            setattr(engine_args, field_name, value)
            applied[field_name] = value

    if applied:
        logger.info(
            "Applied benchmark-tuned EngineArgs defaults for Qwen3-4B "
            "(input_len=%s, output_len=%s): %s",
            getattr(args, "input_len", None),
            getattr(args, "output_len", None),
            applied,
        )

    return applied


def _should_bypass_enforce_eager_for_cuda_runtime() -> bool:
    try:
        import torch
    except Exception:
        return False

    if not torch.cuda.is_available():
        return False

    if getattr(torch.version, "cuda", None) is None:
        return False

    if getattr(torch.version, "hip", None) is not None:
        return False

    try:
        device_name = torch.cuda.get_device_name(0).lower()
    except Exception:
        return True

    return "nvidia" in device_name or "geforce" in device_name or "rtx" in device_name


def _patch_engine_args_for_cuda_graph() -> None:
    if not _should_bypass_enforce_eager_for_cuda_runtime():
        return

    from vllm.engine.arg_utils import EngineArgs

    if getattr(EngineArgs.create_model_config, "_fl_cuda_graph_patched", False):
        return

    original_create_model_config = EngineArgs.create_model_config

    def create_model_config_with_cuda_graph(engine_args):
        original_enforce_eager = engine_args.enforce_eager
        if original_enforce_eager:
            logger.info(
                "Ignoring enforce_eager in EngineArgs on NVIDIA CUDA so "
                "CUDA Graph remains enabled."
            )
            engine_args.enforce_eager = False
        try:
            return original_create_model_config(engine_args)
        finally:
            engine_args.enforce_eager = original_enforce_eager
    create_model_config_with_cuda_graph._fl_cuda_graph_patched = True
    EngineArgs.create_model_config = create_model_config_with_cuda_graph


def _patch_engine_args_for_benchmark_defaults() -> None:
    if not _should_bypass_enforce_eager_for_cuda_runtime():
        return

    from vllm.engine.arg_utils import EngineArgs

    if getattr(EngineArgs, "_fl_benchmark_from_cli_args_patched", False):
        return

    original_from_cli_args = EngineArgs.from_cli_args.__func__

    def from_cli_args_with_benchmark_defaults(cls, args):
        engine_args = original_from_cli_args(cls, args)
        _apply_qwen3_4b_benchmark_defaults(engine_args, args)
        return engine_args

    EngineArgs.from_cli_args = classmethod(from_cli_args_with_benchmark_defaults)
    EngineArgs._fl_benchmark_from_cli_args_patched = True


def __getattr__(name):
    if name == "distributed":
        import importlib
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _patch_transformers_compat():
    """Patch transformers compatibility for ALLOWED_LAYER_TYPES and tokenizer."""
    import transformers.configuration_utils as cfg
    if not hasattr(cfg, "ALLOWED_LAYER_TYPES"):
        cfg.ALLOWED_LAYER_TYPES = getattr(
            cfg, "ALLOWED_ATTENTION_LAYER_TYPES", ()
        )


def register():
    """Register the FL platform."""
    from vllm_fl.utils import get_op_config as _get_op_config

    _patch_transformers_compat()
    _patch_engine_args_for_scheduler_defaults()

    # Model-specific platform patches
    from vllm_fl.patches.glm_moe_dsa import apply_platform_patches as glm5_platform
    glm5_platform()

    multiproc_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
    if multiproc_method is None:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    _get_op_config()
    return "vllm_fl.platform.PlatformFL"


def register_model():
    """Register the FL model."""
    _patch_engine_args_for_benchmark_defaults()
    _patch_engine_args_for_cuda_graph()
    _patch_engine_args_for_scheduler_defaults()

    from vllm import ModelRegistry
    import vllm.model_executor.models.qwen3_next as qwen3_next_module

    # Register Qwen3.5 MoE config
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        from vllm_fl.configs.qwen3_5_moe import Qwen3_5MoeConfig
        _CONFIG_REGISTRY["qwen3_5_moe"] = Qwen3_5MoeConfig
    except Exception as e:
        logger.error(f"Register Qwen3.5 MoE config error: {str(e)}")

    # Register Qwen3Next model
    try:
        from vllm_fl.models.qwen3_next import Qwen3NextForCausalLM  # noqa: F401

        qwen3_next_module.Qwen3NextForCausalLM = Qwen3NextForCausalLM
        logger.warning(
            "Qwen3NextForCausalLM has been patched to use vllm_fl.models.qwen3_next, "
            "original vLLM implementation is overridden"
        )

        ModelRegistry.register_model(
            "Qwen3NextForCausalLM",
            "vllm_fl.models.qwen3_next:Qwen3NextForCausalLM"
        )
    except Exception as e:
        logger.error(f"Register Qwen3Next model error: {str(e)}")

    # Register Qwen3.5 MoE model
    try:
        ModelRegistry.register_model(
            "Qwen3_5MoeForConditionalGeneration",
            "vllm_fl.models.qwen3_5:Qwen3_5MoeForConditionalGeneration"
        )
    except Exception as e:
        logger.error(f"Register Qwen3.5 MoE model error: {str(e)}")

    # Register MiniCPMO model
    try:
        ModelRegistry.register_model(
            "MiniCPMO",
            "vllm_fl.models.minicpmo:MiniCPMO"
        )
    except Exception as e:
        logger.error(f"Register MiniCPMO model error: {str(e)}")

    # Register Kimi-K2.5 model
    try:
        ModelRegistry.register_model(
            "KimiK25ForConditionalGeneration",
            "vllm_fl.models.kimi_k25:KimiK25ForConditionalGeneration",
        )
    except Exception as e:
        logger.error(f"Register KimiK25 model error: {str(e)}")

    # Register GLM-5 (GlmMoeDsa) model
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        from vllm_fl.configs.glm_moe_dsa import GlmMoeDsaConfig
        _CONFIG_REGISTRY["glm_moe_dsa"] = GlmMoeDsaConfig

        from vllm_fl.patches.glm_moe_dsa import apply_model_patches as glm5_model
        glm5_model()

        ModelRegistry.register_model(
            "GlmMoeDsaForCausalLM",
            "vllm_fl.models.glm_moe_dsa:GlmMoeDsaForCausalLM"
        )
    except Exception as e:
        logger.error(f"Register GlmMoeDsa model error: {str(e)}")

    # Register BGE-M3 pooling backport for vLLM 0.13.x
    try:
        ModelRegistry.register_model(
            "BgeM3EmbeddingModel",
            "vllm_fl.models.bge_m3:BgeM3EmbeddingModel",
        )
    except Exception as e:
        logger.error(f"Register BgeM3EmbeddingModel error: {str(e)}")
