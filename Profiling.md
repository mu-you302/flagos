# Qwen3 Attention Stage CUDA Event Profiling

## Summary

实现轻量 profiling：新增 `VLLM_FL_PROFILE_ATTN_STAGES=1`，在 Qwen3-4B 的 `Qwen3Attention.forward` 内用 CUDA event 统计阶段耗时，并周期性打印聚合结果。默认关闭，不改 `benchmarks/flagos_eval/run_benchmark.sh`，只通过环境变量启用。

目标阶段：

- `qkv_proj`
- `qk_norm`
- `rotary_emb`
- `attn`
- 额外记录 `o_proj` 和 `total`，用于解释剩余耗时

## Key Changes

- 新增 `vllm_fl/patches/qwen3_attn_profile.py`：
  - 提供 `apply_qwen3_attention_stage_profile_patch()`。
  - 仅当 `VLLM_FL_PROFILE_ATTN_STAGES=1` 时 patch upstream `vllm.model_executor.models.qwen3.Qwen3Attention.forward`。
  - patch 后的 forward 逻辑保持原始语义：
    `qkv_proj -> split -> q_norm/k_norm -> rotary_emb -> self.attn -> o_proj`。
  - 每个阶段前后使用 `torch.cuda.Event(enable_timing=True)`。
  - 每次 forward 结束后 synchronize 一次，只在 profiling 开启时发生。

- 在 `vllm_fl/__init__.py::register_model()` 中调用 patch：
  - 放在已有 model registration/patch 附近。
  - patch 函数内部自行判断 env，默认 no-op。
  - 使用 `_fl_stage_profile_patched` guard 防止重复 patch。

- 聚合器行为：
  - 以进程内全局 dict 聚合：`count/sum/min/max`。
  - 按 `num_tokens` 分 bucket，例如 `tokens=1024/4096/6144/other`。
  - 默认每 `VLLM_FL_PROFILE_ATTN_STAGES_LOG_EVERY=100` 次 attention forward 打印一次。
  - 默认跳过前 `VLLM_FL_PROFILE_ATTN_STAGES_WARMUP=10` 次，避免 CUDA Graph warmup/首次 kernel 编译污染。
  - 日志格式为单行 CSV-like，方便 grep：
    `ATTN_STAGE_PROFILE bucket=1024 count=100 qkv_proj_ms=... qk_norm_ms=... rotary_ms=... attn_ms=... o_proj_ms=... total_ms=...`

- 环境变量接口：
  - `VLLM_FL_PROFILE_ATTN_STAGES=1`: 启用。
  - `VLLM_FL_PROFILE_ATTN_STAGES_LOG_EVERY=100`: 每多少次 forward 输出一次。
  - `VLLM_FL_PROFILE_ATTN_STAGES_WARMUP=10`: 跳过前多少次 forward。
  - `VLLM_FL_PROFILE_ATTN_STAGES_BUCKETS=1024,4096,6144`: token bucket，默认这三个。

## Implementation Details

- `Qwen3Attention.forward` 的 profiled 版本直接复制当前 vLLM 0.13 原始 forward，并只插入计时代码。
- `total` 从进入 forward 到 `o_proj` 结束。
- `attn` 阶段只统计 `self.attn(q, k, v)`，不包含 `o_proj`。
- 对非 CUDA、profiling disabled、或 `torch.cuda.is_available()` 为 false 的情况，完全使用原始 forward。
- 异常路径不吞异常；若阶段内报错，原异常继续抛出。
- 日志使用 `logging.getLogger(__name__).info(...)`，不写文件，避免 benchmark 产物污染。

## Usage

运行 benchmark 时不改脚本，只加环境变量：

```bash
VLLM_FL_PROFILE_ATTN_STAGES=1 \
VLLM_FL_PROFILE_ATTN_STAGES_LOG_EVERY=100 \
VLLM_FL_PROFILE_ATTN_STAGES_WARMUP=10 \
CUDA_VISIBLE_DEVICES=2 \
bash benchmarks/flagos_eval/run_benchmark.sh
```

也可以只测单个场景时加同样 env。

## Test Plan

- 静态检查：
  - `python -m py_compile vllm_fl/patches/qwen3_attn_profile.py vllm_fl/__init__.py`

- 功能 smoke：
  - 不设置 env 跑 `test_vllm.py`，确认无额外日志、输出正常。
  - 设置 `VLLM_FL_PROFILE_ATTN_STAGES=1 VLLM_FL_PROFILE_ATTN_STAGES_LOG_EVERY=1` 跑 `test_vllm.py`，确认出现 `ATTN_STAGE_PROFILE` 日志。
  - 确认 profiling 日志包含 `qkv_proj/qk_norm/rotary_emb/attn/o_proj/total`。

- 性能使用：
  - 对 `1k/4k/6k` benchmark 各跑一次 profiling。
  - 用日志里的 `rotary_emb_ms / total_ms` 判断 RoPE 融合收益上限。
  - 若 `rotary_emb` 占 attention block `>=5%` 或端到端估算 `>=1%`，再进入融合实现评估。

## Assumptions

- 目标模型路径仍是 Qwen3-4B，实际模型类是 upstream `vllm.model_executor.models.qwen3.Qwen3Attention`。
- profiling 是诊断工具，不用于正式性能成绩。
- 因 CUDA event 每次 forward 后需要 synchronize，开启后 benchmark 总耗时会变慢，性能结论只看阶段比例，不看端到端 throughput。
