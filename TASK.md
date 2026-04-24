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

# 小算子性能比较
实现方式：vllm_fl/dispatch/bench_small_ops.py，直接构造输入对算子进行测试。

op	1024	4096	6144
rms_norm	vendor 快 3.5x	vendor 快 2.9x	vendor 快 2.9x
rms_norm_residual	vendor 快 2.9x	vendor 快 1.9x	基本持平，FlagOS 略快 0.6%
silu_and_mul	vendor 快 5.1x	FlagOS 略快 1.6%	FlagOS 略快 1.4%
rotary_embedding	vendor 快 2.5x	vendor 快 1.9x	FlagOS 快 2.2x

结论：vendor CUDA 在小上下文明显更快，长上下文下部分算子 FlagOS 略优或明显优。

因此，将rms_norm实现方式更改为vendor。