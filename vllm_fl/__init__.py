# Copyright (c) 2025 BAAI. All rights reserved.

import os
import logging

from vllm_fl.utils import get_op_config as _get_op_config

from . import version as version  # PyTorch-style: vllm_fl.version.git_version


logger = logging.getLogger(__name__)


def _resolve_create_engine_config_usage_context(args, kwargs):
    if "usage_context" in kwargs:
        return kwargs["usage_context"]
    if args:
        return args[0]
    return None


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


def _patch_engine_args_for_scheduler_defaults() -> None:
    from vllm.config import SchedulerConfig
    from vllm.engine.arg_utils import EngineArgs

    if getattr(EngineArgs.create_engine_config, "_fl_scheduler_defaults_patched", False):
        return

    original_create_engine_config = EngineArgs.create_engine_config

    def create_engine_config_with_scheduler_metadata(engine_args, *args, **kwargs):
        usage_context = _resolve_create_engine_config_usage_context(args, kwargs)
        user_provided = engine_args.max_num_batched_tokens is not None
        upstream_default = None

        if not user_provided:
            world_size = (
                engine_args.pipeline_parallel_size
                * engine_args.tensor_parallel_size
            )
            default_max_num_batched_tokens, _ = engine_args.get_batch_defaults(world_size)
            upstream_default = default_max_num_batched_tokens.get(
                usage_context,
                SchedulerConfig.DEFAULT_MAX_NUM_BATCHED_TOKENS,
            )

        vllm_config = original_create_engine_config(engine_args, *args, **kwargs)
        scheduler_config = vllm_config.scheduler_config
        setattr(
            scheduler_config,
            "_fl_user_provided_max_num_batched_tokens",
            user_provided,
        )
        setattr(
            scheduler_config,
            "_fl_upstream_default_max_num_batched_tokens",
            upstream_default,
        )
        return vllm_config

    create_engine_config_with_scheduler_metadata._fl_scheduler_defaults_patched = True
    EngineArgs.create_engine_config = create_engine_config_with_scheduler_metadata


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
