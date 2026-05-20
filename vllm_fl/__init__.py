# Copyright (c) 2025 BAAI. All rights reserved.

import json
import os
import logging
from typing import Any

from . import version as version  # PyTorch-style: vllm_fl.version.git_version


logger = logging.getLogger(__name__)


_QWEN3_4B_BENCHMARK_DEFAULTS = {
    (1024, 1024): {
        "max_model_len": 2056,
        "max_num_batched_tokens": 16384,
        # "max_num_seqs": 512,
        "max_num_seqs": 384,
    },
    (4096, 1024): {
        "max_model_len": 5128,
        "max_num_batched_tokens": 12288,
        "max_num_seqs": 256,
    },
    (6144, 1024): {
        "max_model_len": 7176,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 128,
    },
}

_DEFAULT_DRAFT_MODEL_PATH = "/data/Qwen3-0.6B"


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


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _get_optional_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return value


def _get_speculative_config_from_env() -> dict[str, Any] | None:
    raw_config = os.environ.get("VLLM_FL_SPECULATIVE_CONFIG")
    if raw_config:
        config = json.loads(raw_config)
        if not isinstance(config, dict):
            raise ValueError("VLLM_FL_SPECULATIVE_CONFIG must be a JSON object.")
        return config

    if not _truthy_env("VLLM_FL_ENABLE_DRAFT_SPECULATIVE"):
        return None

    return {
        "model": os.environ.get("VLLM_FL_DRAFT_MODEL", _DEFAULT_DRAFT_MODEL_PATH),
        "num_speculative_tokens": int(
            os.environ.get("VLLM_FL_NUM_SPECULATIVE_TOKENS", "4")
        ),
    }


def _apply_speculative_config_default(engine_args, args) -> bool:
    if not _should_bypass_enforce_eager_for_cuda_runtime():
        return False

    if getattr(engine_args, "speculative_config", None) is not None:
        return False

    if not _get_qwen3_4b_benchmark_defaults(args):
        return False

    speculative_config = _get_speculative_config_from_env()
    if speculative_config is None:
        return False

    speculative_config = dict(speculative_config)
    speculative_config.setdefault("model", _DEFAULT_DRAFT_MODEL_PATH)
    speculative_config.setdefault("num_speculative_tokens", 4)
    setattr(engine_args, "speculative_config", speculative_config)
    logger.info("Applied vllm_fl speculative config: %s", speculative_config)
    return True


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

def _apply_cuda_quantization_default(engine_args) -> bool:
    if not _should_bypass_enforce_eager_for_cuda_runtime():
        return False

    if not hasattr(engine_args, "quantization"):
        return False

    if getattr(engine_args, "quantization", None) is not None:
        return False

    engine_args.quantization = "fp8"
    logger.info("Setting quantization to fp8 by default on NVIDIA CUDA.")
    return True

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
        _apply_cuda_quantization_default(engine_args)
        _apply_speculative_config_default(engine_args, args)
        return engine_args

    EngineArgs.from_cli_args = classmethod(from_cli_args_with_benchmark_defaults)
    EngineArgs._fl_benchmark_from_cli_args_patched = True

def _apply_cuda_flaggems_attention_defaults() -> None:
    if not _should_bypass_enforce_eager_for_cuda_runtime():
        return

    os.environ.setdefault("USE_FLAGGEMS", "1")
    if (
        "VLLM_FL_FLAGOS_WHITELIST" not in os.environ
        and "VLLM_FL_FLAGOS_BLACKLIST" not in os.environ
    ):
        os.environ["VLLM_FL_FLAGOS_WHITELIST"] = "attention_backend"


def _patch_engine_args_for_draft_model_spec_decode() -> None:
    from vllm.engine.arg_utils import EngineArgs

    if getattr(EngineArgs, "_fl_draft_model_spec_decode_patched", False):
        return

    original_check_feature_supported = EngineArgs._check_feature_supported

    def check_feature_supported_with_draft_model(self, model_config):
        speculative_config = getattr(self, "speculative_config", None)
        method = (
            speculative_config.get("method")
            if isinstance(speculative_config, dict)
            else getattr(speculative_config, "method", None)
        )
        if method == "draft_model":
            if isinstance(speculative_config, dict):
                patched_config = dict(speculative_config)
                patched_config.pop("method", None)
                self.speculative_config = patched_config
                try:
                    return original_check_feature_supported(self, model_config)
                finally:
                    self.speculative_config = speculative_config

        return original_check_feature_supported(self, model_config)

    EngineArgs._check_feature_supported = check_feature_supported_with_draft_model
    EngineArgs._fl_draft_model_spec_decode_patched = True


def _patch_speculative_config_for_draft_model() -> None:
    import vllm.config.speculative as speculative_module
    from vllm.config import ModelConfig
    from vllm.config.speculative import SpeculativeConfig

    if getattr(SpeculativeConfig, "_fl_draft_model_patched", False):
        return

    original_post_init = SpeculativeConfig.__post_init__

    def _looks_like_plain_draft_model(config) -> bool:
        if config.model is None:
            return False
        if config.method == "draft_model":
            return True
        if config.method is not None:
            return False
        model_name = str(config.model).lower()
        return not any(marker in model_name for marker in ("eagle", "medusa"))

    def _init_plain_draft_model(config) -> None:
        if config.target_model_config is None:
            raise ValueError("target_model_config must be present for draft_model")
        if config.target_parallel_config is None:
            raise ValueError("target_parallel_config must be present for draft_model")

        config.method = "draft_model"
        config.prompt_lookup_max = 0
        config.prompt_lookup_min = 0

        target_model_config = config.target_model_config
        draft_quantization = _get_optional_env("VLLM_FL_DRAFT_QUANTIZATION")
        config.draft_model_config = ModelConfig(
            model=config.model,
            runner="draft",
            tokenizer=target_model_config.tokenizer,
            tokenizer_mode=target_model_config.tokenizer_mode,
            trust_remote_code=target_model_config.trust_remote_code,
            allowed_local_media_path=target_model_config.allowed_local_media_path,
            allowed_media_domains=target_model_config.allowed_media_domains,
            dtype=target_model_config.dtype,
            seed=target_model_config.seed,
            revision=config.revision,
            code_revision=config.code_revision,
            tokenizer_revision=target_model_config.tokenizer_revision,
            spec_target_max_model_len=target_model_config.max_model_len,
            quantization=draft_quantization,
            enforce_eager=target_model_config.enforce_eager,
            max_logprobs=target_model_config.max_logprobs,
            hf_overrides=SpeculativeConfig.hf_config_override,
            config_format=target_model_config.config_format,
        )

        if config.num_speculative_tokens is None:
            raise ValueError(
                "num_speculative_tokens must be provided for draft_model "
                "speculative decoding."
            )

        if config.speculative_token_tree is None:
            config.speculative_token_tree = str(
                [(i + 1) * (0,) for i in range(config.num_speculative_tokens)]
            )

        config.draft_tensor_parallel_size = (
            SpeculativeConfig._verify_and_get_draft_tp(
                config.target_parallel_config,
                config.draft_tensor_parallel_size,
                config.draft_model_config.hf_config,
            )
        )
        config.draft_model_config.max_model_len = (
            SpeculativeConfig._maybe_override_draft_max_model_len(
                config.max_model_len,
                config.draft_model_config.max_model_len,
                target_model_config.max_model_len,
            )
        )
        config.draft_parallel_config = SpeculativeConfig.create_draft_parallel_config(
            config.target_parallel_config,
            config.draft_tensor_parallel_size,
        )

    def post_init_with_plain_draft_model(self):
        if _looks_like_plain_draft_model(self):
            _init_plain_draft_model(self)
            return self
        return original_post_init(self)

    SpeculativeConfig.__post_init__ = post_init_with_plain_draft_model
    speculative_module.SpeculativeConfig.__post_init__ = post_init_with_plain_draft_model
    SpeculativeConfig._fl_draft_model_patched = True

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
    _apply_cuda_flaggems_attention_defaults()

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
    # _patch_speculative_config_for_draft_model()
    # _patch_engine_args_for_draft_model_spec_decode()
    _patch_engine_args_for_benchmark_defaults()
    _patch_engine_args_for_cuda_graph()

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
