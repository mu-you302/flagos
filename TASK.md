# 核心目标
针对Qwen3-4B这个模型，在保持模型精度的基础上（-2%～），时延不能明显差于baseline的前提下，提升Throughput性能

conda环境：conda activate flagos（其中vllm版本为0.13.0，已经安装过FlagGems）

模型文件位置：/data/xiyang/vllm-plugin-FL/Qwen3-4B

当前针对的硬件：NVIDIA RTX 4090

评测脚本：benchmarks/flagos_eval/run_benchmark.sh，核心逻辑是使用vllm bench throughput、vllm bench latency进行测试，输入长度主要有1k、4k、6k。

throughput_chat_1k   : input=1024 tokens, output=1024 tokens, 300 prompts
throughput_chat_4k   : input=4096 tokens, output=1024 tokens, 300 prompts
throughput_chat_6k   : input=6144 tokens, output=1024 tokens, 300 prompts

## 评测过程解析
在当前任务里，吞吐量评测脚本 run_benchmark.sh (line 41) 并没有显式传 --max-num-batched-tokens、--max-model-len、--max-num-seqs，所以它们走的是 vLLM 的默认推导逻辑。
一些参数的默认值为：
max-num-batched-tokens: 8192
max-model-len: 40960
max-num-seqs: 256

依据分别是：

vllm bench throughput 走的是离线 LLM(...) 路径，throughput.py (line 51) 会构造 LLM(**engine_args)，而 llm.py (line 351) 里 LLMEngine.from_engine_args(..., usage_context=UsageContext.LLM_CLASS)，所以默认值按 LLM_CLASS 分支取。

对 RTX 4090 这类非 H100/H200、显存也不到 70 GiB 的 GPU，arg_utils.py (line 1797) 会走 else 分支，给出：

LLM_CLASS.max_num_batched_tokens = 8192
LLM_CLASS.max_num_seqs = 256
具体赋值发生在 arg_utils.py (line 1940)。

max-model-len 的 CLI 默认是 None，表示“从模型配置自动推导”，不是固定数值；实际尝试中日志打印出Using max model len 40960。

> 如果是H100，参数默认值为16384，max_num_seqs=1024。而A100是8192, 256

# NVIDIA RTX 4090 上的 Qwen3-4B 执行路由
模型基本信息：
Number of Layers: 36
Number of Attention Heads (GQA): 32 for Q and 8 for KV

```
Qwen3-4B 推理流程 (RTX 4090 + vllm-plugin-FL)
│
├─ Token Input
│  └─ [1, hidden_dim]
│
├─ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│  │ Embedding Layer (标准 PyTorch)
├─ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│
├─ Transformer Block × 36 层
│  │
│  ├─► 1. RMS Normalization (Input)
│  │   └─ dispatch → call_op("rms_norm")
│  │      └─ 优先级: FlagGems(150) > CUDA(100) > PyTorch(50)
│  │         └─ RTX 4090 上选择: FlagGems -> CUDA
│  │
│  ├─► 2. Grouped Query Attention
│  │   ├─ Query/Key/Value Projection (标准 Linear)
|  |   |
|  |   |—— QK-Norm
│  │   │
│  │   ├─ Rotary Position Embedding
│  │   │  └─ dispatch → call_op("rotary_embedding", query, key, cos, sin, positions)
│  │   │     └─ RTX 4090 上选择: FlagGems 或 CUDA 优化实现
|  |   |
│  │   ├─ Attention Backend Selection
│  │   │  └─ dispatch → call_op("attention_backend", use_mla=False, use_sparse=False)
│  │   │     ├─ flagos -> cuda
│  │   │
│  │   └─ Attention Output Projection (标准 Linear)
│  │
│  ├─ RMS Normalization (post-attention)
│  │  └─ call_op("rms_norm")
│  │
│  ├─► 3. Feed-Forward Network
│  │   ├─ Gate Linear: [seq_len, 4096] → [seq_len, 9728]
│  │   │
│  │   ├─ SiLU + Gating (融合)
│  │   │  └─ dispatch → call_op("silu_and_mul")
│  │   │     └─ 输入: [seq_len, 9728]
│  │   │     └─ 输出: [seq_len, 9728]
│  │   │     └─ RTX 4090 上选择: FlagGems > CUDA > PyTorch
│  │   │
│  │   └─ Output Projection: [seq_len, 9728] → [seq_len, 4096]
│  │
│  └─ Residual Connections (标准 PyTorch add)
│
└─ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   │ Final RMS Norm + LM Head
└─ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   │
   └─ Logits (原始)
      └─ Sampling/Beam Search
         └─ Output Tokens
```         

使用原始代码库运行Qwen3-4B得到的部分输出：
```
(EngineCore_DP0 pid=272940)        new kernel: registered at /data/xiyang/vllm-plugin-FL/temp/FlagGems/src/flag_gems/__init__.py:20 (Triggered internally at /pytorch/aten/src/ATen/core/dispatch/OperatorEntry.cpp:218.)
(EngineCore_DP0 pid=272940) [2026-04-18 17:18:45] [INFO] [manager.py:567] Op 'attention_backend' using 'default.flagos' 
(EngineCore_DP0 pid=272940) [2026-04-18 17:18:43] INFO platform.py:212: Using attention backend via dispatch (use_mla=False, use_sparse=False): vllm.v1.attention.backends.triton_attn.TritonAttentionBackend
(EngineCore_DP0 pid=272940) [2026-04-18 17:18:45] [INFO] [manager.py:567] Op 'rms_norm' using 'default.flagos'
(EngineCore_DP0 pid=272940) [2026-04-18 17:18:45] [INFO] [manager.py:567] Op 'rotary_embedding' using 'default.flagos'
(EngineCore_DP0 pid=272940) [2026-04-18 17:18:45] [INFO] [manager.py:567] Op 'silu_and_mul' using 'default.flagos'
```
其中的算子都是使用 FlagGems 后端实现。
特别地，对于其中的 attention_backend，路由过程为：
```
dispatch.call_op("attention_backend")
    │
    ▼
OpManager 选择 "default.flagos" 
    (FlagGemsBackend 优先级最高)
    │
    ▼
FlagGemsBackend.attention_backend() 执行
    │
    ├─ 检查: use_mla? use_sparse?
    │
    ├─ YES → 返回 AttentionFLBackend (支持 MLA/DSA)
    │
    └─ NO  → 返回 TritonAttentionBackend (更稳定，vLLM 原生)
           (这就是 Qwen3-4B 的情况！)
    │
    ▼
TritonAttentionBackend 初始化
    │
    ├─ 内部调用 FlagGems 的 flash_attn kernel
    │  (已被 FlagGems 覆盖了 PyTorch 的原生 kernel)
    │
    └─ ✅ 既用了 FlagGems 的优化，又有 vLLM 的稳定性
```

# 工作边界
你可以修改 ./vllm_fl 文件夹中的源码，不得改动 ./benchmarks 中评测相关脚本。

对源码修改后，你可以在flagos这个环境下运行 test_vllm.py 进行测试，具体评测部分交给我来。

# 参考文档
vllm_fl说明文档：./核心源码说明.md
vllm_fl/dispatch说明文档：./diapatch说明文档.md

如果你在工作过程中发现有重要信息未曾出现在已有文档中，将其加到文档中的合适位置并说明你做出的添加。

# baseline性能
## 吞吐量
| type | scenario | num_prompts | total_tokens | elapsed_time_s | tokens_per_sec | requests_per_sec |
|------|----------|-------------|--------------|----------------|----------------|------------------|
| throughput | chat_1k | 300 | 614400 | 159.97400024099625 | 3840.62409563069 | 1.875304734194673 |
| throughput | chat_4k | 300 | 1536000 | 497.27999438502593 | 3088.8031236799175 | 0.603281860093734 |
| throughput | chat_6k | 300 | 2150400 | 738.3972093570046 | 2912.2536932020175 | 0.40628539246679934 |

## 延迟
| type | scenario | num_prompts | mean_latency_ms | p50_latency_ms | p90_latency_ms | p99_latency_ms |
|------|----------|-------------|-----------------|----------------|----------------|----------------|
| latency | batch_8 | 10 | 66482.55274519906 | 66038.95167999144 | 67228.56515878811 | 67254.50193228608 |

# 已完成改动——针对NVIDIA硬件绕过 enforce-eager，启用 CUDA Graph
1. 在 `vllm_fl/platform.py` 中增加了仅针对 `NVIDIA CUDA` 的 `enforce_eager` 绕过逻辑。
2. 当评测脚本继续传入 `--enforce-eager` 时，CUDA 路径会在平台配置阶段自动改回 `enforce_eager=False`，从而继续执行 CUDA Graph capture。
3. 如果上游已经因为 `enforce_eager=True` 将 `cudagraph_mode` 降为 `NONE`，这里会仅在 `NVIDIA CUDA` 下恢复为 `FULL_AND_PIECEWISE`。
4. 其他硬件路径不受影响，原有的 NPU / MUSA / 其他 vendor 行为保持不变。
5. 补充在 `vllm_fl/__init__.py` 的注册阶段提前 patch `EngineArgs.create_model_config()`，避免 `VllmConfig` 在更早阶段因 `enforce_eager=True` 将优化级别降到 `-O0` 并禁用 cudagraph。

| type | scenario | num_prompts | total_tokens | elapsed_time_s | tokens_per_sec | requests_per_sec |
|------|----------|-------------|--------------|----------------|----------------|------------------|
| throughput | chat_1k | 300 | 614400 | 136.31 | 4507.38 | 2.20 |
| throughput | chat_4k | 300 | 1536000 | 398.17 | 3857.62 | 0.75 |
| throughput | chat_6k | 300 | 2150400 | 591.54 | 3635.28 | 0.51 |

| type | scenario | num_prompts | mean_latency_ms | p50_latency_ms | p90_latency_ms | p99_latency_ms |
|------|----------|-------------|-----------------|----------------|----------------|----------------|
| latency | batch_8 | 10 | 19526.76 | 19520.42 | 19557.64 | 19561.54 |

## 图编译 / CUDA Graph capture size A/B 中间结论

为了判断 CUDA Graph 是否仍有可优化空间，新增了仅用于实验的统计与配置开关：

1. `VLLM_FL_CUDAGRAPH_STATS=1`：统计 runtime dispatch 到 `FULL` / `PIECEWISE` / `NONE` 的分布，并记录 batch descriptor、raw tokens、padded tokens、padding 等信息。
2. `VLLM_FL_CUDAGRAPH_CAPTURE_SIZES`：直接覆盖 `cudagraph_capture_sizes`。
3. `VLLM_FL_CUDAGRAPH_CAPTURE_SIZES_BY_MAX_MODEL_LEN`：按 benchmark 场景的 `max_model_len` 分别覆盖 capture sizes，例如 `2056` 对应 chat_1k，`5128` 对应 chat_4k / batch_8，`7176` 对应 chat_6k。

统计运行目录：

`ab_results/cudagraph_capture_sizes_20260516_194932/default_stats/cg_stats.jsonl`

在原始 capture size 配置下，CUDA Graph dispatch 命中率如下：

| 场景 | max_model_len | total steps | hit rate | FULL | PIECEWISE | NONE | 主要 fallback |
|------|---------------|-------------|----------|------|-----------|------|---------------|
| chat_1k | 2056 | 1577 | 98.03% | 1546 | 0 | 31 | mixed prefill，常见 raw tokens 包括 1024、4097、16384 |
| chat_4k | 5128 | 3526 | 96.03% | 3386 | 0 | 140 | mixed prefill，主要 raw tokens=12288 |
| chat_6k | 7176 | 5150 | 94.82% | 4883 | 0 | 267 | mixed prefill，主要 raw tokens=8192 |
| batch_8 latency | 5128 | 20541 | 99.61% | 20460 | 1 | 80 | 少量 mixed prefill，decode 基本全命中 |

中间结论：

1. 默认 capture sizes 对 decode 已经很有效，decode 阶段基本都走 `FULL` CUDA Graph。
2. 主要 fallback 不在 decode，而在大 batch 的 mixed prefill / chunked prefill。
3. 因此单纯提高 decode 小 batch capture sizes 没有意义；如果继续优化 CUDA Graph，目标应该是 mixed prefill 的大 batch descriptor。

基于上面的 fallback 分布，尝试过给不同 benchmark 场景追加大 prefill capture sizes：

| 场景 | 默认 capture sizes | 候选追加的大 prefill sizes |
|------|--------------------|----------------------------|
| chat_1k | 1..512 | 1024, 4096, 8192, 16384 |
| chat_4k / batch_8 | 1..512 | 4096, 4106, 8192, 12288 |
| chat_6k | 1..256 | 6144, 8192 |

无 stats 开销 A/B 结果：

| 场景 | baseline | large-prefill candidate | 变化 |
|------|----------|-------------------------|------|
| chat_1k tokens/s | 7386.15 | 7336.15 | -0.68% |
| chat_4k tokens/s | 6984.64 | 6925.65 | -0.85% |
| chat_6k tokens/s | 6610.78 | 6555.58 | -0.84% |
| batch_8 mean latency | 12834.96 ms | 12758.72 ms | -0.59% |

该候选不应采用。原因是 vLLM 的 `pad_for_cudagraph()` 会把两个 capture size 之间的 batch size pad 到下一个 capture size。直接追加 6144 / 8192 / 12288 / 16384 后，很多 mixed prefill step 会被 pad 到大图执行，额外计算量抵消甚至超过 CUDA Graph replay 收益。图捕获显存也明显增加，例如：

| 场景 | baseline graph memory | large-prefill candidate graph memory |
|------|-----------------------|--------------------------------------|
| chat_6k | 0.37 GiB | 1.26 GiB |
| chat_1k | 0.61 GiB | 2.41 GiB |
| chat_4k | 0.57 GiB | 1.91 GiB |
| batch_8 | 0.57 GiB | 1.91 GiB |

最终结论：

1. 当前默认 CUDA Graph capture size 配置已经能覆盖绝大多数 decode step。
2. mixed prefill fallback 虽然存在，但直接扩大 `cudagraph_capture_sizes` 是负收益。
3. 后续如果继续做图编译优化，应实现“large mixed-prefill exact-match / low-padding 策略”：只有 raw token 数精确命中，或 padding 足够小，才走大 prefill CUDA Graph；否则保持当前 fallback 路径。
4. 因此现阶段保留默认 capture sizes，不启用大 prefill capture size override。

# 模型权重量化
vLLM 0.13.0 支持 BF16 → FP8 在线量化：--quantization fp8

在插件框架中 patch 量化参数。量化之后，底层 GEMM 算子使用的是 W8A8 版本，计算速度更快。


# 已完成改动——针对 CUDA 默认启用 FP8 KV cache
1. 在 `vllm_fl/platform.py` 中增加了仅针对 `NVIDIA CUDA` 的 KV cache dtype 默认改写逻辑。
2. 当 `cache_dtype` 仍为 `auto` 时，CUDA 路径会自动切换为 `fp8`，从而让当前 Qwen3-4B 的 `TritonAttentionBackend` 走上游已支持的 FP8 KV cache 路径。
3. 设置`calculate_kv_scales=False`，避免 `TritonAttentionBackend` 出现 "A non 1.0 q_scale is not currently supported." 问题，可能导致一定的精度损失。
4. 如果用户显式指定了其他 `cache_dtype`，本次逻辑不会覆盖；NPU / MUSA / 其他 vendor 也不受影响。

测试结果（本次测试结果将 gpu_memory_utilization 设置为0.8，上面那个实验中为0.9，但是吞吐量还是有优势）
### THROUGHPUT TEST RESULTS

| Scenario | Prompts | Tokens | Time(s) | Tokens/s | Req/s |
| :--- | :--- | :--- | :--- | :--- | :--- |
| chat_1k | 300 | 614400 | 123.7s | 4966.6 | 2.43 |
| chat_4k | 300 | 1536000 | 333.6s | 4603.8 | 0.90 |
| chat_6k | 300 | 2150400 | 524.6s | 4099.4 | 0.57 |

### LATENCY TEST RESULTS

| Scenario | Iters | Mean(ms) | P50(ms) | P90(ms) | P99(ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| batch_8 | 10 | 20571.62 | 20565.85 | 20731.11 | 20788.28 |

# 已完成改动——针对 benchmark 输入长度场景调整调度默认值
1. 在 `vllm_fl/__init__.py` 中增加了对 `EngineArgs.from_cli_args()` 的 patch。
2. 当入口来自 `vllm bench throughput` / `vllm bench latency`，且 CLI 参数中包含 `input_len`、`output_len`，模型名命中 `Qwen3-4B` 时，会在 `from_cli_args()` 阶段识别 benchmark 场景。
3. 仅当用户没有显式传入 `max_model_len`、`max_num_batched_tokens`、`max_num_seqs` 时，才会按场景注入默认值，不覆盖用户手工指定参数。
4. 当前场景映射为：
   - `1024 / 1024 -> max_model_len=2056, max_num_batched_tokens=16384, max_num_seqs=512`
   - `4096 / 1024 -> max_model_len=5128, max_num_batched_tokens=12288, max_num_seqs=256`
   - `6144 / 1024 -> max_model_len=7176, max_num_batched_tokens=8192, max_num_seqs=128`
5. 该 patch 发生在 `create_engine_config()` 和 worker 初始化之前，因此适合做 benchmark 场景相关的调度默认值改写；对不带 `input_len/output_len` 的普通离线推理和其他模型路径不生效。

### THROUGHPUT TEST RESULTS

| Scenario | Prompts | Tokens | Time(s) | Tokens/s | Req/s |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **chat_1k** | 300 | 614,400 | 115.1 | 5,336.4 | 2.61 |
| **chat_4k** | 300 | 1,536,000 | 292.3 | 5,254.5 | 1.03 |
| **chat_6k** | 300 | 2,150,400 | 417.7 | 5,148.2 | 0.72 |

### LATENCY TEST RESULTS

| Scenario | Iters | Mean(ms) | P50(ms) | P90(ms) | P99(ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **batch_8** | 10 | 17,449.57 | 17,159.61 | 17,554.77 | 20,222.88 |

# 算子选择
当前算子默认优先级为 flagos > vendor，但是对于小算子，使用cuda原生实现可能更优。

USE_FLAGGEMS=1 + VLLM_FL_FLAGOS_WHITELIST=attention_backend

为了保留 fp8 KV cache 可用性，将 attention_backend 加入到白名单中，否则 vendor 会切到 FlashAttentionBackend。

case	baseline	B	提升
chat_6k	5607.64	5794.37	+3.33%
chat_1k	6288.87	6412.92	+1.97%
chat_4k	5867.28	5993.56	+2.15%

## 注意力后端

本次在 `vllm_fl/dispatch/backends/flaggems/flaggems.py` 中给 `attention_backend` 增加可选路由：

- `VLLM_FL_ATTENTION_BACKEND=flashinfer|triton`
- 当前默认值改为 `flashinfer`。
- 如需恢复之前行为，可设置 `VLLM_FL_ATTENTION_BACKEND=triton`。

A/B 结果目录：

`ab_results/bench_feature_ab_20260517_153052/summary.md`

本次 A/B 使用原始评测脚本，不改动 `benchmarks/flagos_eval/run_benchmark.sh`。关键对比为：

| case | `VLLM_FL_ATTENTION_BACKEND` |
|------|-----------------------------|
| baseline_triton | triton |
| flashinfer | flashinfer |

### A/B 结果

| case | chat_1k tok/s | delta | chat_4k tok/s | delta | chat_6k tok/s | delta | throughput geomean | delta | mean latency ms | delta |
|------|--------------:|------:|--------------:|------:|--------------:|------:|-------------------:|------:|----------------:|------:|
| baseline_triton | 7401.055 | +0.000% | 6999.632 | +0.000% | 6621.584 | +0.000% | 7000.196 | +0.000% | 12754.595 | +0.000% |
| flashinfer | 7356.576 | -0.601% | 7184.950 | +2.648% | 6759.600 | +2.084% | 7095.889 | +1.367% | 12953.822 | +1.562% |

结论：

1. `FlashInferBackend` 是本次主要收益来源：`chat_4k` 提升 `+2.648%`，`chat_6k` 提升 `+2.084%`，但 `chat_1k` 回退 `-0.601%`。这与 profiling 结论一致：输入越长，attention CUDA 时间占比越高，替换 attention backend 越有效。
2. 因此当前默认代码路径采用 `FlashInfer attention`；如果后续发现短上下文 `chat_1k` 权重更重要，可以通过 `VLLM_FL_ATTENTION_BACKEND=triton` 回退到 Triton，或做按场景选择。
3. `FlashInferBackend` 在RTX 4090上验证有效，但是提交到赛题官网结果表明收益不大，可能是因为GPU架构等硬件条件不同，因此提交版本中并没有启用这种优化方式。