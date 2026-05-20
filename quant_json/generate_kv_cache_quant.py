#!/usr/bin/env python3
"""Generate static C8 KV-cache quantization parameters for Qwen3.

The output is intentionally separated from the model directory.  To make vLLM
load the generated parameters later, copy/symlink the generated files into the
model directory and use the generated index patch as the model safetensors
index.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import rotate_half


PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-channel C8 KV-cache scale/offset tensors."
    )
    parser.add_argument(
        "--model-path",
        default="/data/xiyang/vllm-plugin-FL/Qwen3-4B",
        help="Path to the local Qwen3 model directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="quant_json",
        help="Directory where quantization files will be written.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=32,
        help="Number of calibration prompts to run.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=1024,
        help="Maximum calibration sequence length.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Calibration device, for example cuda:0 or cpu.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Model dtype used during calibration.",
    )
    parser.add_argument(
        "--param-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help="Dtype used to store generated scale/offset tensors.",
    )
    parser.add_argument(
        "--scheme",
        choices=("asymmetric", "symmetric"),
        default="asymmetric",
        help="Affine quantization scheme for int8 KV cache.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to sample calibration prompts.",
    )
    parser.add_argument(
        "--keep-proxy",
        action="store_true",
        help="Do not unset proxy environment variables before dataset download.",
    )
    parser.add_argument(
        "--hf-endpoint",
        default="https://hf-mirror.com",
        help="HF endpoint used for dataset/model metadata downloads.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Pass trust_remote_code=True to transformers loaders.",
    )
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def clear_proxy_env() -> None:
    for name in PROXY_ENV_VARS:
        os.environ.pop(name, None)


def _safe_load_dataset(dataset_name: str, *args: Any, split: str, **kwargs: Any):
    from datasets import load_dataset

    try:
        return load_dataset(dataset_name, *args, split=split, **kwargs)
    except Exception as exc:
        print(f"[warn] skipped dataset {dataset_name!r}: {type(exc).__name__}: {exc}")
        return None


def collect_calibration_prompts(num_samples: int, seed: int) -> list[str]:
    """Collect task-style prompts aligned with benchmarks/flagos_eval/run_eval.sh."""
    rng = random.Random(seed)
    prompts: list[str] = []

    dataset_specs = [
        ("gsm8k", ("main",), "train", ("question",)),
        ("openai/openai_humaneval", tuple(), "test", ("prompt",)),
        ("google-research-datasets/mbpp", ("sanitized",), "train", ("prompt", "text")),
        ("lukaemon/bbh", ("boolean_expressions",), "test", ("input", "target")),
        ("juletxara/mgsm", ("zh",), "test", ("question", "answer")),
        ("mgsm", ("zh",), "test", ("question", "answer")),
    ]

    per_dataset = max(1, math.ceil(num_samples / len(dataset_specs)))
    for dataset_name, configs, split, fields in dataset_specs:
        dataset = _safe_load_dataset(
            dataset_name,
            *configs,
            split=f"{split}[:{max(per_dataset * 3, per_dataset)}]",
        )
        if dataset is None:
            continue
        rows = list(dataset)
        rng.shuffle(rows)
        for row in rows[:per_dataset]:
            parts = [str(row[field]) for field in fields if field in row and row[field]]
            if parts:
                prompts.append("\n".join(parts))

    fallback = [
        "Solve the following math problem step by step: If a train travels 120 km in 2 hours, what is its speed?",
        "Write a Python function that returns the nth Fibonacci number.",
        "Question: Is the statement true? not (true and false). Answer:",
        "请解答这个数学题：小明有12个苹果，分给3个朋友，每人几个？",
        "Explain why attention KV cache improves autoregressive decoding speed.",
    ]
    while len(prompts) < num_samples:
        prompts.append(fallback[len(prompts) % len(fallback)])
    rng.shuffle(prompts)
    return prompts[:num_samples]


def init_stats(model: torch.nn.Module) -> tuple[dict[str, dict[str, torch.Tensor]], list[Any]]:
    stats: dict[str, dict[str, torch.Tensor]] = {}
    handles = []

    def update_range(
        name: str,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        # key/value shape: [batch, num_kv_heads, seq_len, head_dim]
        k_min = key_states.detach().float().amin(dim=(0, 2)).cpu()
        k_max = key_states.detach().float().amax(dim=(0, 2)).cpu()
        v_min = value_states.detach().float().amin(dim=(0, 2)).cpu()
        v_max = value_states.detach().float().amax(dim=(0, 2)).cpu()
        if name not in stats:
            stats[name] = {
                "k_min": k_min,
                "k_max": k_max,
                "v_min": v_min,
                "v_max": v_max,
            }
            return
        current = stats[name]
        current["k_min"] = torch.minimum(current["k_min"], k_min)
        current["k_max"] = torch.maximum(current["k_max"], k_max)
        current["v_min"] = torch.minimum(current["v_min"], v_min)
        current["v_max"] = torch.maximum(current["v_max"], v_max)

    def make_hook(name: str):
        def hook(module, args, kwargs):
            hidden_states = args[0] if args else kwargs["hidden_states"]
            position_embeddings = (
                args[1] if len(args) > 1 else kwargs["position_embeddings"]
            )
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)

            key_states = module.k_norm(
                module.k_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            key_states = (key_states * cos) + (rotate_half(key_states) * sin)

            update_range(name, key_states, value_states)

        return hook

    for name, module in model.named_modules():
        if (
            hasattr(module, "k_proj")
            and hasattr(module, "v_proj")
            and hasattr(module, "k_norm")
            and hasattr(module, "head_dim")
            and name.endswith("self_attn")
        ):
            handles.append(module.register_forward_pre_hook(make_hook(name), with_kwargs=True))
    if not handles:
        raise RuntimeError("No Qwen3 self_attn modules were found.")
    return stats, handles


def make_scale_offset(
    min_value: torch.Tensor,
    max_value: torch.Tensor,
    scheme: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scheme == "symmetric":
        max_abs = torch.maximum(min_value.abs(), max_value.abs())
        scale = torch.clamp(max_abs / 127.0, min=1.0e-6)
        offset = torch.zeros_like(scale)
    else:
        scale = torch.clamp((max_value - min_value) / 255.0, min=1.0e-6)
        offset = -128.0 - min_value / scale
    return scale.to(dtype=dtype).contiguous(), offset.to(dtype=dtype).contiguous()


def build_quant_description(model_path: Path) -> dict[str, Any]:
    description: dict[str, Any] = {
        "version": "1.0.0",
        "model_quant_type": "FLOAT",
        "kv_cache_type": "C8",
    }
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as f:
            index = json.load(f)
        for name in sorted(index.get("weight_map", {})):
            if name.endswith(".weight"):
                description[name] = "FLOAT"
    return description


def build_index_patch(
    model_path: Path,
    output_dir: Path,
    tensor_names: list[str],
    tensor_file_name: str,
) -> None:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        return
    with index_path.open() as f:
        index = json.load(f)
    weight_map = dict(index.get("weight_map", {}))
    for name in tensor_names:
        weight_map[name] = tensor_file_name
    patched = {
        "metadata": dict(index.get("metadata", {})),
        "weight_map": dict(sorted(weight_map.items())),
    }
    # Keep metadata conservative.  Some loaders ignore total_size, and the
    # original value is still a lower bound if the small C8 shard is added.
    patch_path = output_dir / "model.safetensors.index.c8_patch.json"
    with patch_path.open("w") as f:
        json.dump(patched, f, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    if not args.keep_proxy:
        clear_proxy_env()
    if args.hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")

    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = collect_calibration_prompts(args.num_samples, args.seed)
    with (output_dir / "calibration_prompts.json").open("w") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    print(f"[info] collected {len(prompts)} calibration prompts")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype(args.dtype),
        device_map={"": args.device},
        trust_remote_code=args.trust_remote_code,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()

    stats, handles = init_stats(model)
    try:
        with torch.inference_mode():
            for idx, prompt in enumerate(prompts, start=1):
                encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_seq_len,
                )
                encoded = {k: v.to(args.device) for k, v in encoded.items()}
                _ = model(**encoded, use_cache=False)
                print(
                    f"[info] calibrated sample {idx}/{len(prompts)} "
                    f"(tokens={encoded['input_ids'].shape[-1]})"
                )
    finally:
        for handle in handles:
            handle.remove()

    if not stats:
        raise RuntimeError("Calibration finished without collecting KV stats.")

    param_dtype = torch_dtype(args.param_dtype)
    assert isinstance(param_dtype, torch.dtype)
    tensors: dict[str, torch.Tensor] = {}
    layer_summaries: dict[str, dict[str, float]] = {}

    for attn_name in sorted(stats, key=lambda x: int(x.split(".")[-2])):
        s = stats[attn_name]
        k_scale, k_offset = make_scale_offset(
            s["k_min"], s["k_max"], args.scheme, param_dtype
        )
        v_scale, v_offset = make_scale_offset(
            s["v_min"], s["v_max"], args.scheme, param_dtype
        )
        tensors[f"{attn_name}.k_proj.kv_cache_scale"] = k_scale
        tensors[f"{attn_name}.k_proj.kv_cache_offset"] = k_offset
        tensors[f"{attn_name}.v_proj.kv_cache_scale"] = v_scale
        tensors[f"{attn_name}.v_proj.kv_cache_offset"] = v_offset
        layer_summaries[attn_name] = {
            "k_scale_min": float(k_scale.float().min()),
            "k_scale_max": float(k_scale.float().max()),
            "v_scale_min": float(v_scale.float().min()),
            "v_scale_max": float(v_scale.float().max()),
        }

    tensor_file_name = "kv_cache_scales.safetensors"
    save_file(tensors, output_dir / tensor_file_name)

    quant_description = build_quant_description(model_path)
    with (output_dir / "quant_model_description.json").open("w") as f:
        json.dump(quant_description, f, indent=2, sort_keys=True)

    build_index_patch(
        model_path=model_path,
        output_dir=output_dir,
        tensor_names=sorted(tensors),
        tensor_file_name=tensor_file_name,
    )

    manifest = {
        "model_path": str(model_path),
        "num_samples": len(prompts),
        "max_seq_len": args.max_seq_len,
        "scheme": args.scheme,
        "param_dtype": args.param_dtype,
        "tensor_file": tensor_file_name,
        "num_tensors": len(tensors),
        "layers": layer_summaries,
    }
    with (output_dir / "kv_cache_quant_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"[done] wrote C8 KV-cache quant files to {output_dir}")


if __name__ == "__main__":
    main()
