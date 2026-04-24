# Copyright (c) 2026 BAAI. All rights reserved.

"""Micro-benchmark FlagOS vs vendor implementations for small dispatch ops.

This module is intentionally kept under ``vllm_fl`` instead of ``benchmarks`` so
the benchmark evaluation scripts remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from vllm_fl.dispatch import SelectionPolicy, get_default_manager, policy_context


DEFAULT_MODEL_CONFIG = "/data/xiyang/vllm-plugin-FL/Qwen3-4B/config.json"
DEFAULT_TOKENS = (1024, 4096, 6144)
DEFAULT_OPS = ("rms_norm", "silu_and_mul", "rotary_embedding")
DEFAULT_BACKENDS = ("flagos", "vendor")
BACKEND_TOKENS = {
    "flagos": "flagos",
    "vendor": "vendor:cuda",
}


@dataclass
class ModelShape:
    hidden_size: int
    intermediate_size: int
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float


@dataclass
class RMSNormObj:
    weight: torch.Tensor
    variance_epsilon: float


@dataclass
class OpCase:
    op_name: str
    scenario: str
    tokens: int
    shape: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


def _dtype_from_name(name: str) -> torch.dtype:
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def _load_model_shape(path: str) -> ModelShape:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    return ModelShape(
        hidden_size=int(cfg["hidden_size"]),
        intermediate_size=int(cfg["intermediate_size"]),
        head_dim=int(cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])),
        num_attention_heads=int(cfg["num_attention_heads"]),
        num_key_value_heads=int(cfg["num_key_value_heads"]),
        max_position_embeddings=int(cfg.get("max_position_embeddings", 40960)),
        rope_theta=float(cfg.get("rope_theta", 1000000.0)),
        rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
    )


def _make_rope_cache(
    max_positions: int,
    rotary_dim: int,
    theta: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )
    positions = torch.arange(max_positions, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _make_case(
    op_name: str,
    tokens: int,
    shape: ModelShape,
    dtype: torch.dtype,
    device: torch.device,
    include_rms_residual: bool,
) -> list[OpCase]:
    scenario = f"ctx_{tokens}"

    if op_name == "rms_norm":
        weight = torch.randn(shape.hidden_size, dtype=dtype, device=device)
        obj = RMSNormObj(weight=weight, variance_epsilon=shape.rms_norm_eps)
        x = torch.randn(tokens, shape.hidden_size, dtype=dtype, device=device)
        cases = [
            OpCase(
                op_name=op_name,
                scenario=scenario,
                tokens=tokens,
                shape=f"x=[{tokens},{shape.hidden_size}], residual=None",
                args=(obj, x, None),
                kwargs={},
            )
        ]
        if include_rms_residual:
            x_res = torch.randn(tokens, shape.hidden_size, dtype=dtype, device=device)
            residual = torch.randn(tokens, shape.hidden_size, dtype=dtype, device=device)
            cases.append(
                OpCase(
                    op_name=op_name,
                    scenario=f"{scenario}_residual",
                    tokens=tokens,
                    shape=f"x/residual=[{tokens},{shape.hidden_size}]",
                    args=(obj, x_res, residual),
                    kwargs={},
                )
            )
        return cases

    if op_name == "silu_and_mul":
        fused_width = shape.intermediate_size * 2
        x = torch.randn(tokens, fused_width, dtype=dtype, device=device)
        return [
            OpCase(
                op_name=op_name,
                scenario=scenario,
                tokens=tokens,
                shape=f"x=[{tokens},{fused_width}] -> out=[{tokens},{shape.intermediate_size}]",
                args=(None, x),
                kwargs={},
            )
        ]

    if op_name == "rotary_embedding":
        q = torch.randn(
            tokens,
            shape.num_attention_heads,
            shape.head_dim,
            dtype=dtype,
            device=device,
        )
        k = torch.randn(
            tokens,
            shape.num_key_value_heads,
            shape.head_dim,
            dtype=dtype,
            device=device,
        )
        cache_len = max(tokens, shape.max_position_embeddings)
        cos, sin = _make_rope_cache(
            cache_len, shape.head_dim // 2, shape.rope_theta, dtype, device
        )
        positions = torch.arange(tokens, dtype=torch.long, device=device)
        return [
            OpCase(
                op_name=op_name,
                scenario=scenario,
                tokens=tokens,
                shape=(
                    f"q=[{tokens},{shape.num_attention_heads},{shape.head_dim}], "
                    f"k=[{tokens},{shape.num_key_value_heads},{shape.head_dim}]"
                ),
                args=(None, q, k, cos, sin, positions, False, True),
                kwargs={},
            )
        ]

    raise ValueError(f"Unsupported op: {op_name}")


def _resolve_backend(op_name: str, backend: str) -> Callable[..., Any]:
    policy = SelectionPolicy.from_dict(
        prefer=backend,
        strict=True,
        per_op_order={op_name: [BACKEND_TOKENS[backend]]},
        allow_vendors={"cuda"} if backend == "vendor" else None,
    )
    with policy_context(policy):
        return get_default_manager().resolve(op_name)


def _event_times_ms(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    warmup: int,
    iterations: int,
) -> list[float]:
    with torch.inference_mode():
        for _ in range(warmup):
            fn(*args, **kwargs)
        torch.cuda.synchronize()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        for i in range(iterations):
            starts[i].record()
            fn(*args, **kwargs)
            ends[i].record()
        torch.cuda.synchronize()

    return [starts[i].elapsed_time(ends[i]) for i in range(iterations)]


def _summarize(times_ms: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(times_ms),
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "std_ms": statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
    }


def _benchmark_case(
    case: OpCase,
    backend: str,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    fn = _resolve_backend(case.op_name, backend)
    times = _event_times_ms(fn, case.args, case.kwargs, warmup, iterations)
    result = {
        "op": case.op_name,
        "backend": backend,
        "scenario": case.scenario,
        "tokens": case.tokens,
        "shape": case.shape,
        "warmup": warmup,
        "iterations": iterations,
    }
    result.update(_summarize(times))
    return result


def _format_ms(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.4f}"


def _print_backend_rows(rows: list[dict[str, Any]]) -> None:
    print("\nPer-backend results")
    print(
        "op,scenario,backend,mean_ms,median_ms,min_ms,max_ms,std_ms,iterations,shape"
    )
    for row in rows:
        if "error" in row:
            print(
                f"{row['op']},{row['scenario']},{row['backend']},ERROR,"
                f"{row['error']}"
            )
            continue
        print(
            f"{row['op']},{row['scenario']},{row['backend']},"
            f"{row['mean_ms']:.4f},{row['median_ms']:.4f},"
            f"{row['min_ms']:.4f},{row['max_ms']:.4f},"
            f"{row['std_ms']:.4f},{row['iterations']},{row['shape']}"
        )


def _print_comparison(rows: list[dict[str, Any]]) -> None:
    by_case: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if "error" in row:
            continue
        by_case.setdefault((row["op"], row["scenario"]), {})[row["backend"]] = row

    print("\nFlagOS vs vendor comparison")
    print("op,scenario,flagos_mean_ms,vendor_mean_ms,vendor_ms/flagos_ms,winner")
    for (op_name, scenario), backend_rows in sorted(by_case.items()):
        flagos = backend_rows.get("flagos")
        vendor = backend_rows.get("vendor")
        flagos_ms = flagos["mean_ms"] if flagos else None
        vendor_ms = vendor["mean_ms"] if vendor else None
        if flagos_ms and vendor_ms:
            ratio = vendor_ms / flagos_ms
            winner = "flagos" if flagos_ms < vendor_ms else "vendor"
            ratio_s = f"{ratio:.4f}"
        else:
            ratio_s = "n/a"
            winner = "n/a"
        print(
            f"{op_name},{scenario},{_format_ms(flagos_ms)},"
            f"{_format_ms(vendor_ms)},{ratio_s},{winner}"
        )


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "op",
        "scenario",
        "tokens",
        "backend",
        "mean_ms",
        "median_ms",
        "min_ms",
        "max_ms",
        "std_ms",
        "warmup",
        "iterations",
        "shape",
        "error",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark FlagOS and CUDA vendor implementations of rms_norm, "
            "silu_and_mul, and rotary_embedding with Qwen3-style shapes."
        )
    )
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--tokens", type=int, nargs="+", default=list(DEFAULT_TOKENS))
    parser.add_argument("--ops", nargs="+", default=list(DEFAULT_OPS), choices=DEFAULT_OPS)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(DEFAULT_BACKENDS),
        choices=DEFAULT_BACKENDS,
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--no-rms-residual",
        action="store_true",
        help="Skip the fused residual RMSNorm case.",
    )
    parser.add_argument("--csv", default="", help="Optional CSV output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    dtype = _dtype_from_name(args.dtype)
    device = torch.device(args.device)
    shape = _load_model_shape(args.model_config)

    print(
        "Benchmark config: "
        f"model_config={args.model_config}, dtype={args.dtype}, "
        f"tokens={args.tokens}, warmup={args.warmup}, "
        f"iterations={args.iterations}"
    )
    print(
        "Model shape: "
        f"hidden={shape.hidden_size}, intermediate={shape.intermediate_size}, "
        f"q_heads={shape.num_attention_heads}, kv_heads={shape.num_key_value_heads}, "
        f"head_dim={shape.head_dim}"
    )

    rows: list[dict[str, Any]] = []
    for op_name in args.ops:
        for tokens in args.tokens:
            for backend in args.backends:
                cases = _make_case(
                    op_name,
                    tokens,
                    shape,
                    dtype,
                    device,
                    include_rms_residual=not args.no_rms_residual,
                )
                for case in cases:
                    try:
                        row = _benchmark_case(
                            case,
                            backend,
                            warmup=args.warmup,
                            iterations=args.iterations,
                        )
                    except Exception as exc:
                        row = {
                            "op": case.op_name,
                            "backend": backend,
                            "scenario": case.scenario,
                            "tokens": case.tokens,
                            "shape": case.shape,
                            "warmup": args.warmup,
                            "iterations": args.iterations,
                            "error": repr(exc),
                        }
                    rows.append(row)
            torch.cuda.empty_cache()

    _print_backend_rows(rows)
    _print_comparison(rows)
    if args.csv:
        _write_csv(args.csv, rows)
        print(f"\nWrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
