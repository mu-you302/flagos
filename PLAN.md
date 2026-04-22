# Qwen3-4B 吞吐优化方案（4090 / vLLM 0.13 / 当前 benchmark）

## 摘要
- 默认路线继续保持当前已验证有效的组合：`CUDA Graph + fp8 KV cache + calculate_kv_scales=False + TritonAttentionBackend`。
- 首轮只落地两类低风险改动：`batched token budget` 提升，以及 NVIDIA 上 `RMSNorm / SwiGLU / RoPE` 的 dispatch 重排。
- 第二轮再做可控 A/B：`async scheduling`、`prefix caching`、`FlagGems 全局 patch 收缩`。
- 长线方案才考虑 Qwen3 定制融合核，因为它工程量大，但从模型结构看有明确上限空间。

## 关键改动
### 第一优先级：直接落地
- 在 `PlatformFL` 中新增 Qwen3-4B 专用调度默认值：
  - 条件：`vendor == NVIDIA`，`model_type == qwen3`，模型名命中 `Qwen3-4B`，`kv_cache_dtype in {auto, fp8}`，且用户未显式设置 `max_num_batched_tokens`。
  - 行为：将 `max_num_batched_tokens` 从当前默认 `8192` 提升到 `16384`。
  - 失败回退：若初始化或首轮 profile 出现 OOM，则自动退回 `12288`；仍失败再退回现有默认。
  - 保持 `max_num_seqs` 不变，避免把吞吐变化和排队策略变化混在一起。
- 调整 `dispatch/config/nvidia.yaml` 的默认优先级：
  - `silu_and_mul: vendor -> flagos -> reference`
  - `rms_norm: vendor -> flagos -> reference`
  - `rotary_embedding: vendor -> flagos -> reference`
  - `attention_backend: flagos -> vendor -> reference` 保持不变
- 原因和判断：
  - Qwen3-4B 每层有 `input RMSNorm + post-attn RMSNorm + q_norm + k_norm + SwiGLU`，36 层下这些点算子非常密集。
  - 当前 `vendor/cuda` 实现直接走 vLLM 自带 CUDA custom op；而 FlagGems 这几项在当前环境更像 Python/Triton 包装层，默认不应放在 NVIDIA 首位。
  - `attention_backend` 不能直接切到 vendor 默认，因为 4090 上 FlashAttention 路径无法保留 fp8 KV，这会和现有最大收益项冲突。

### 第二优先级：逐项 A/B，不作为首轮默认
- 增加可选开关 `FLAGOS_ENABLE_ASYNC_SCHED=1`：
  - 仅在 `single GPU / no PP / no speculative decoding / no structured outputs` 时生效。
  - 实现位置仍在 `PlatformFL`，只改 `async_scheduling`，不碰其他调度参数。
- 增加可选开关 `FLAGOS_BENCH_NO_PREFIX_CACHE=1`：
  - 仅用于离线吞吐 benchmark。
  - 行为：在用户未显式关闭该开关前提下，将 `enable_prefix_caching=False`。
  - 原因：当前评测是随机 prompt，无共享前缀，prefix cache 基本没有命中收益，只会增加管理开销。
- 收缩 `worker.py` 里的 `flag_gems.enable(all ops)`：
  - NVIDIA 默认不再“全量接管” ATen CUDA op。
  - 白名单先对齐 FlagGems 自身 NVIDIA 默认配置，只保留 `softmax`、`cumsum` 一类已知低风险 patch，其余依赖 OOT dispatch。
  - 这一项单独 A/B，避免与 dispatch 重排互相掩盖效果。

### 第三优先级：高投入结构优化
- 新增 Qwen3 专用融合路径：`q_norm + k_norm + rotary`。
- 实现方式：
  - 在 `vllm_fl/ops` 增加新 OOT op；
  - 在 `dispatch/backends/vendor/cuda/impl` 增加 CUDA/Triton 实现；
  - 在插件初始化时 patch Qwen3 attention 前处理路径，仅对 `model_type=qwen3` 且非 MLA 路径启用。
- 目标：
  - 减少 Q/K 两次 RMSNorm、一次 RoPE、若干中间 tensor 的 kernel launch 和读写；
  - 这是最符合 Qwen3 结构特征的长期优化方向，但不应放进首轮。

## 接口变化
- 不改 benchmark 脚本，不改用户 Python 调用方式。
- 仅新增两个可选环境变量作为第二轮 A/B 开关：
  - `FLAGOS_ENABLE_ASYNC_SCHED=1`
  - `FLAGOS_BENCH_NO_PREFIX_CACHE=1`

## 测试与验收
- 冒烟检查：
  - `test_vllm.py` 正常跑通；
  - 单次生成确认 CUDA Graph、fp8 KV、dispatch 选中的 backend 与预期一致。
- 基准复测：
  - throughput：`1k/1k`、`4k/1k`、`6k/1k`
  - latency：`batch=8, 4k/1k`
- 验收基线：
  - 以 `TASK.md` 当前结果为基线：约 `4967 / 4604 / 4099 tok/s`，平均延迟约 `20.57s`
  - 首轮成功标准：`4k/1k`、`6k/1k` 吞吐继续提升，准确率下降不超过约 `2%`，latency 不出现明显回退
- 执行顺序：
  1. 只做 `max_num_batched_tokens=16384` + dispatch 重排并复测
  2. 若收益不足，再单独打开 `async scheduling`
  3. 再单独测试 `prefix caching off`
  4. 最后再测试 `FlagGems 全局 patch 收缩`

## 假设与默认
- 当前最优默认组合仍然是 `TritonAttention + fp8 KV`，不是 `FlashAttention + bf16 KV`。
- 不修改 `benchmarks/` 目录，所有行为都从 `vllm_fl` 注入。
- 第一轮只追求低风险吞吐收益，不把 benchmark 定向技巧和结构性重写混在一起。
- 若 `16384` 在 4090 上不稳定，默认回退到 `12288`，但保留 dispatch 重排。
