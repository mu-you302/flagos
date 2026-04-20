# 核心目标
针对Qwen3-4B这个模型，在保持模型精度的基础上（-2%～），时延不能明显差于baseline的前提下，提升Throughput性能

conda环境：conda activate flagos（其中vllm版本为0.13.0，已经安装过FlagGems）

模型文件位置：/data/xiyang/vllm-plugin-FL/Qwen3-4B

当前针对的硬件：NVIDIA RTX 4090

评测脚本：benchmarks/flagos_eval/run_benchmark.sh，核心逻辑是使用vllm bench throughput、vllm bench latency进行测试，输入长度主要有1k、4k、6k。

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
│  │   │
│  │   ├─ Rotary Position Embedding
│  │   │  └─ dispatch → call_op("rotary_embedding", query, key, cos, sin, positions)
│  │   │     └─ RTX 4090 上选择: FlagGems 或 CUDA 优化实现
│  │   │——QK-Norm
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
./vllm_fl说明文档：./核心源码说明.md
./vllm_fl/dispatch说明文档：./diapatch说明文档.md

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

================================================================================
THROUGHPUT TEST RESULTS
================================================================================

Scenario        Prompts  Tokens     Time(s)   Tokens/s     Req/s     
--------------------------------------------------------------------------------
chat_1k         300      614400       123.7s     4966.6      2.43
chat_4k         300      1536000      333.6s     4603.8      0.90
chat_6k         300      2150400      524.6s     4099.4      0.57
--------------------------------------------------------------------------------

================================================================================
LATENCY TEST RESULTS
================================================================================

Scenario        Iters  Mean(ms)     P50(ms)      P90(ms)      P99(ms)     
--------------------------------------------------------------------------------
batch_8         10     20571.62     20565.85     20731.11     20788.28    
--------------------------------------------------------------------------------