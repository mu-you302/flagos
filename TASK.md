# 核心目标
针对Qwen3-4B这个模型，在保持模型精度的基础上（-2%～），时延不能明显差于baseline的前提下，提升Throughput性能

评测脚本：benchmarks/flagos_eval/run_benchmark.sh，核心是使用vllm bench throughput、vllm bench latency进行测试，输入长度主要有1k、4k、6k。

# 参考文档
vllm_fl部分说明文档：./核心源码说明.md
vlm_fl/dispatch 部分说明文档：./diapatch说明文档.md

如果你在工作过程中发现有重要信息未曾出现在已有文档中，将其加到文档中的合适位置并说明你做出的添加。

# baseline情况
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
