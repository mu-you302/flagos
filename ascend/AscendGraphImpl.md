# Ascend Graph FULL_DECODE_ONLY 实现交接记录

本文记录当前 `vllm_fl` 插件中针对 Ascend 910C 实现 FULL_DECODE_ONLY 图编译的改动。目标是在不走 Inductor 的前提下，复用 vLLM 上层 `CUDAGraphMode` / `CudagraphDispatcher` 抽象，在 Ascend 底层执行 `torch.npu.NPUGraph` capture/replay，并保证 `test_vllm.py` 输出正确。

## 1. 当前结论

本轮已经实现并验证：

- Ascend NPU 上默认启用 `CUDAGraphMode.FULL_DECODE_ONLY`。
- Ascend 图模式强制使用 `CompilationMode.NONE`，避开当前环境下不稳定的 `torch.compile` / Inductor 路径。
- Ascend graph capture sizes 默认收敛到 `[1, 2, 4, 8]`，避免上游默认 bucket 过多导致显存压力过大。
- NPU capture 使用本地 graph capture 上下文，不再误走上游 CUDA graph capture。
- `torch_npu._npu_paged_attention` 在 FULL_DECODE_ONLY replay 前通过 NPU graph task update 热更新 `seq_lens` 等运行时参数，避免 capture dummy metadata 固化导致生成结果错误。
- 吞吐 benchmark 中暴露的 `graph_task_update_end` 507000 崩溃已修复：NPU graph replay 阶段必须先 `replay()` 将图任务入队，再执行 task update 并 record event 放行。
- `test_vllm.py` 在 `flagos` conda 环境下通过，输出与无图基线一致。

验证命令：

```bash
VLLM_PLUGINS=fl /root/miniconda3/envs/flagos/bin/python test_vllm.py
```

关键日志：

```text
cudagraph_mode=<CUDAGraphMode.FULL_DECODE_ONLY: (2, 0)>
cudagraph_capture_sizes=[1, 2, 4, 8]
max_cudagraph_capture_size=8
mode=<CompilationMode.NONE: 0>
Capturing CUDA graphs (decode, FULL): 4/4
Graph capturing finished in 1 secs, took 0.94 GiB
```

验证输出：

```text
Prompt: 'Hello, my name is', Generated text: " Xiaoyu, and I'm a student at"
```

注意：日志中的 `CUDA graphs` 是 vLLM 上层历史命名，Ascend 底层实际执行的是 `torch.npu.NPUGraph`。

## 2. 设计边界

本轮只落地 FULL_DECODE_ONLY：

- 不支持 FULL prefill/mixed graph。
- 不支持 PIECEWISE。
- 不启用 Inductor。
- 不启用 vllm-ascend 中更完整的 OOT compiler / fusion pass 体系。
- 只处理当前 Qwen3-4B decode path 依赖的 Ascend paged attention 动态参数更新。

这样做的原因是前序尝试已经确认：

- `FULL` / mixed prefill-decode graph 会固化不适合当前路径的 metadata，输出存在偏移。
- 即使 `FULL_DECODE_ONLY` 能 capture，如果 `seq_lens` 等 attention metadata 不在 replay 前更新，也会生成错误 token。
- vLLM 默认 capture bucket 过多时，Ascend 910C 上显存占用过高，实际会 OOM。

## 3. 改动文件总览

| 文件 | 作用 |
| --- | --- |
| `vllm_fl/__init__.py` | 启动期 patch `EngineArgs`，让 Ascend 运行时绕过 `enforce_eager`；设置 Triton Ascend 环境变量 |
| `vllm_fl/platform.py` | Ascend 平台图模式决策：`FULL_DECODE_ONLY + CompilationMode.NONE + [1,2,4,8]` |
| `vllm_fl/worker/model_runner.py` | NPU 走本地 graph capture；初始化 graph attention 参数表 |
| `vllm_fl/compilation/graph.py` | 扩展 `GraphWrapper`，实现 NPU graph capture/replay 和 attention task update |
| `vllm_fl/dispatch/backends/vendor/ascend/impl/attention.py` | capture 阶段注册 paged attention workspace、event、handle 和参数 |
| `tests/unit_tests/test_engine_args_patch.py` | 覆盖 Ascend `EngineArgs` graph patch |
| `tests/unit_tests/test_platform_graph_config.py` | 覆盖 Ascend 平台图模式默认配置 |
| `tests/unit_tests/compilation/test_graph.py` | 覆盖 `GraphParams` 和 NPU paged attention update |

## 4. 启动期 EngineArgs patch

文件：`vllm_fl/__init__.py`

新增 Ascend 运行时检测：

```python
def _should_bypass_enforce_eager_for_ascend_runtime() -> bool:
    try:
        import torch
        import torch_npu
    except Exception:
        return False

    return bool(hasattr(torch, "npu") and torch.npu.is_available())
```

新增统一入口：

```python
def _get_graph_preferred_platform_runtime() -> str | None:
    if _should_bypass_enforce_eager_for_ascend_runtime():
        return "Ascend NPU"
    if _should_bypass_enforce_eager_for_cuda_runtime():
        return "NVIDIA CUDA"
    return None
```

`_patch_engine_args_for_graph()` 会 patch `EngineArgs.create_model_config()`：

- 如果用户或测试脚本传入 `enforce_eager=True`，创建 `model_config` 前临时置为 `False`。
- `create_model_config()` 返回后恢复原始 `engine_args.enforce_eager`。
- 这样可以让 vLLM config 阶段保留 graph capture 能力，同时不破坏外层对象状态。

`register()` 中增加：

```python
if _should_bypass_enforce_eager_for_ascend_runtime():
    os.environ.setdefault("TRITON_ALL_BLOCKS_PARALLEL", "1")
```

这是为了避开 Triton Ascend backend 在 KV cache 大张量初始化时的 grid 限制。

## 5. 平台图模式决策

文件：`vllm_fl/platform.py`

新增 Ascend capture size 默认值：

```python
ASCEND_DEFAULT_CUDAGRAPH_CAPTURE_SIZES = [1, 2, 4, 8]
```

`PlatformFL.check_and_update_config()` 中 Ascend 分支的最终行为：

```python
compilation_config.mode = CompilationMode.NONE

if compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
    # 用户显式关闭时保持关闭
else:
    if ASCEND_LAUNCH_BLOCKING == "1":
        raise RuntimeError(...)

    if cudagraph_mode in {FULL, FULL_AND_PIECEWISE, PIECEWISE}:
        override to FULL_DECODE_ONLY

    compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    compilation_config.cudagraph_capture_sizes = [1, 2, 4, 8]
    compilation_config.max_cudagraph_capture_size = 8
```

capture sizes 会按 `scheduler_config.max_num_seqs` 过滤。例如 `max_num_seqs=2` 时只保留 `[1, 2]`。

保留的行为：

- 如果用户显式设置 `cudagraph_mode=NONE`，Ascend graph 不会被强行开启。
- 如果 `ASCEND_LAUNCH_BLOCKING=1`，直接报错，因为该环境变量和 NPU graph capture 不兼容。
- data parallel 下原本禁用 graph 的逻辑不再作用于 NPU。

## 6. WorkerRunner 接入

文件：`vllm_fl/worker/model_runner.py`

原逻辑只在 `current_platform.dist_backend == "flagcx"` 时使用本地 NPU graph capture；Ascend 单卡常见是 `hccl`，会误走上游 CUDA graph capture。

已改为：

```python
if current_platform.device_type == "npu" or current_platform.dist_backend == "flagcx":
    @contextmanager
    def graph_capture(device: torch.device):
        ...
else:
    from vllm.distributed.parallel_state import graph_capture
```

在 `_check_and_update_cudagraph_mode()` 中，dispatcher 初始化后为 NPU full graph 初始化参数表：

```python
if current_platform.device_type == "npu" and cudagraph_mode.has_full_cudagraphs():
    reset_graph_params()
    set_graph_params(self.cudagraph_batch_sizes)
```

这里的 `self.cudagraph_batch_sizes` 就是最终要 capture 的 bucket，例如 `[1, 2, 4, 8]`。

## 7. GraphWrapper 扩展

文件：`vllm_fl/compilation/graph.py`

新增 `GraphParams`：

```python
@dataclass
class GraphParams:
    events: dict[int, list[Any]]
    workspaces: dict[int, Any]
    handles: dict[int, list[Any]]
    attn_params: dict[int, list[tuple]]
```

含义：

- `events[size]`：capture 时为每个 attention task 创建的外部 event。
- `workspaces[size]`：`torch_npu._npu_paged_attention` graph update 所需 workspace。
- `handles[size]`：`torch.npu.graph_task_group_end()` 返回的 task handle。
- `attn_params[size]`：capture 时记录的 paged attention 参数。

新增接口：

```python
set_graph_params(cudagraph_capture_sizes)
reset_graph_params()
get_graph_params()
update_graph_params_workspaces(num_tokens, workspace)
update_npu_paged_attention_graph_params(update_stream, forward_context, runtime_shape)
```

capture 阶段：

- NPU 下为 `forward_context` 临时设置 `capturing=True`。
- attention backend 看到该标记后注册 paged attention task 参数。
- capture 完成后恢复原始 `forward_context.capturing`。
- workspace 和输出会转为 weak ref，避免 graph pool 之外持有多余大 tensor。

replay 阶段：

```python
current_platform.torch_device_fn.synchronize()
entry.graph.replay()
if current_platform.device_type == "npu" and self.update_stream is not None:
    update_npu_paged_attention_graph_params(
        self.update_stream,
        forward_context,
        batch_descriptor.num_tokens,
    )
```

重点是 NPU graph task update 的顺序和普通“先改参数再执行”不同：

1. capture 阶段在图内插入 `ExternalEvent.wait(stream)`，让 paged attention task 等待外部 event。
2. replay 阶段先调用 `entry.graph.replay()`，把已捕获的图任务入队。
3. 然后调用 `graph_task_update_begin/end()` 更新已入队 task group 的 runtime 参数。
4. 最后 `event.record(update_stream)` 放行图内等待的 attention task。

如果把 `graph_task_update_begin/end()` 放在 `entry.graph.replay()` 之前，Ascend runtime 找不到可更新的已入队 task group，长吞吐场景会在 `graph_task_update_end()` 报 `507000`。

## 8. Ascend attention graph-safe paged attention

文件：`vllm_fl/dispatch/backends/vendor/ascend/impl/attention.py`

在 `forward_paged_attention()` 里增加 capture 分支。触发条件：

```python
graph_params is not None
and getattr(forward_context, "capturing", False)
and num_tokens in graph_params.attn_params
```

capture 时执行：

1. 如当前 bucket 还没有 workspace，则调用：

```python
torch_npu._npu_paged_attention_get_workspace(...)
```

2. 创建外部 event：

```python
event = torch.npu.ExternalEvent()
event.wait(stream)
event.reset(stream)
```

3. 记录 task 参数：

```python
graph_params.attn_params[num_tokens].append(
    (
        weak_ref_tensors(query),
        weak_ref_tensors(self.key_cache),
        weak_ref_tensors(self.value_cache),
        self.num_kv_heads,
        self.num_heads,
        self.scale,
        attn_metadata.block_tables,
        attn_metadata.seq_lens,
        weak_ref_tensors(output),
    )
)
```

4. 用 graph task group 包住 paged attention：

```python
torch.npu.graph_task_group_begin(stream)
torch_npu._npu_paged_attention(..., workspace=workspace)
handle = torch.npu.graph_task_group_end(stream)
graph_params.handles[num_tokens].append(handle)
```

replay 入队后更新时，`update_npu_paged_attention_graph_params()` 会从当前 `forward_context.attn_metadata[key].seq_lens` 读取运行时 `seq_lens`，再调用：

```python
torch.npu.graph_task_update_begin(update_stream, handle)
torch_npu._npu_paged_attention(..., context_lens=runtime_seq_lens, workspace=workspace)
torch.npu.graph_task_update_end(update_stream)
event.record(update_stream)
```

这是本轮输出正确的核心修复。

## 9. 吞吐 benchmark 后续问题：task update 顺序错误

在 `benchmarks/flagos_eval/run_benchmark.sh` 的吞吐测试中，`chat_4k/chat_6k` 这类长输出场景曾在接近完成时崩溃。典型现象：

```text
Processed prompts: 99%|...| 296/300
RuntimeError: graph_task_update_end ... error code is 507000
EE9999: Update tasks failed, stream_id=42, total=1, success=0, failed=1
```

当时 engine dump 显示已经进入 decode-only graph：

```text
total_num_scheduled_tokens=2
num_scheduled_tokens={299: 1, 298: 1}
cudagraph_mode=<CUDAGraphMode.FULL_DECODE_ONLY: (2, 0)>
cudagraph_capture_sizes=[1, 2, 4, 8]
max_cudagraph_capture_size=8
```

根因不是 capture size、prefix cache 或 KV cache 污染，而是 `GraphWrapper.__call__()` 中的 replay/update 顺序写反了。错误版本是：

```python
current_platform.torch_device_fn.synchronize()
update_npu_paged_attention_graph_params(...)
entry.graph.replay()
```

Ascend task update 机制需要更新 replay 后已经入队、且被 `ExternalEvent` 阻塞的 task group；提前 update 会导致 `graph_task_update_end()` 在 runtime 层失败。

已修复为：

```python
current_platform.torch_device_fn.synchronize()
entry.graph.replay()
update_npu_paged_attention_graph_params(...)
```

对应测试：

```text
tests/unit_tests/compilation/test_graph.py::TestGraphWrapper::test_npu_replay_happens_before_task_update
```

修复后用轻量端到端验证确认：

```bash
VLLM_PLUGINS=fl /root/miniconda3/envs/flagos/bin/python test_vllm.py
```

结果：

```text
Capturing CUDA graphs (decode, FULL): 4/4
Graph capturing finished in 1 secs, took 0.94 GiB
Prompt: 'Hello, my name is', Generated text: " Xiaoyu, and I'm a student at"
```

完整 benchmark 仍建议重新跑一轮，因为原问题是在长时间吞吐测试尾部暴露的；但当前崩溃栈中的直接错误路径已经修复。

## 10. KV cache view 修正

文件：`vllm_fl/dispatch/backends/vendor/ascend/impl/attention.py`

`_get_fused_infer_attention_params()` 中原先 decode/chunked prefill 分支存在硬编码：

```python
key = self.key_cache.view(-1, block_size, 256)
value = self.value_cache.view(-1, block_size, 256)
```

已改为根据模型实际 KV hidden size 计算：

```python
kv_hidden_size = self.num_kv_heads * self.head_size
key = self.key_cache.view(-1, block_size, kv_hidden_size)
value = self.value_cache.view(-1, block_size, kv_hidden_size)
```

这避免在非 256 KV hidden size 模型上 view 形状错误。

## 11. 当前默认参数

在当前 `test_vllm.py` 中显式传入：

```python
max_num_batched_tokens=16384
max_num_seqs=2048
```

如果用户不传这两个参数，vLLM `EngineArgs` 层默认是 `None`，后续按 usage/hardware 推导。当前 Ascend 910C 日志显示单卡显存约 `61.27 GiB`，不满足 vLLM 大显存 `>=70 GiB` 分支，因此 `LLM_CLASS` 默认推导为：

```text
max_num_batched_tokens=8192
max_num_seqs=256
```

插件本轮只改 Ascend graph capture sizes，未改 vLLM 对 `max_num_batched_tokens` / `max_num_seqs` 的通用默认推导。

### prefix caching 默认值说明

吞吐测试日志里出现：

```text
enable_prefix_caching=True
```

这是 vLLM 0.13 当前默认行为，不是本轮 Ascend graph 改动额外打开的。`EngineArgs.enable_prefix_caching` 默认是 `None`，最终落到 `CacheConfig.enable_prefix_caching=True`。因此 benchmark 日志中看到 prefix caching 开启是预期现象。

当前 benchmark 的 `Prefix cache hit rate: 0.0%` 也说明这组固定长度随机 prompt 没有实际命中 prefix cache；它不是 `graph_task_update_end` 507000 的根因。

## 12. 测试情况

已通过的检查：

```bash
/root/miniconda3/envs/flagos/bin/python -m py_compile \
  vllm_fl/__init__.py \
  vllm_fl/platform.py \
  vllm_fl/compilation/graph.py \
  vllm_fl/dispatch/backends/vendor/ascend/impl/attention.py \
  vllm_fl/worker/model_runner.py \
  tests/unit_tests/test_engine_args_patch.py \
  tests/unit_tests/test_platform_graph_config.py \
  tests/unit_tests/compilation/test_graph.py
```

```bash
git diff --check -- \
  vllm_fl/__init__.py \
  vllm_fl/platform.py \
  vllm_fl/compilation/graph.py \
  vllm_fl/dispatch/backends/vendor/ascend/impl/attention.py \
  vllm_fl/worker/model_runner.py \
  tests/unit_tests/test_engine_args_patch.py \
  tests/unit_tests/test_platform_graph_config.py \
  tests/unit_tests/compilation/test_graph.py
```

pytest 未完整运行，原因是 `flagos` 环境没有安装 pytest：

```text
/root/miniconda3/envs/flagos/bin/python: No module named pytest
```

尝试使用 `/root/miniconda3/envs/vllm/bin/python -m pytest` 时，该环境缺少当前项目依赖 `vllm`，无法作为有效验证环境。

新增/调整的测试覆盖点：

- `test_patch_engine_args_for_graph_bypasses_enforce_eager`
- `test_graph_params_initialization_and_reset`
- `test_npu_paged_attention_update_uses_runtime_seq_lens`
- `test_npu_replay_happens_before_task_update`
- `test_ascend_defaults_to_full_decode_only_graph`
- `test_ascend_respects_explicit_cudagraph_none`

本次 benchmark 崩溃修复后的额外验证：

```bash
/root/miniconda3/envs/flagos/bin/python -m py_compile \
  vllm_fl/compilation/graph.py \
  tests/unit_tests/compilation/test_graph.py
```

```bash
VLLM_PLUGINS=fl /root/miniconda3/envs/flagos/bin/python test_vllm.py
```

`flagos` 环境仍未安装 pytest，因此没有在该环境里运行 pytest：

```text
/root/miniconda3/envs/flagos/bin/python: No module named pytest
```

## 13. 已知风险和后续建议

1. `GraphParams` 当前是进程内全局状态，依赖 worker 初始化时按最终 capture sizes 重置。后续如果支持多模型、多 runner 或动态重配，需要检查隔离性。

2. `update_npu_paged_attention_graph_params()` 目前通过 `zip(forward_context.attn_metadata, attn_params, handles, events)` 对齐 layer 顺序，依赖 capture 阶段和 replay 阶段 attention metadata 插入顺序一致。后续如改 attention layer 命名或 metadata 容器，需要重点验证。

3. NPU task update 顺序必须保持为 `synchronize -> replay -> graph_task_update_begin/end -> event.record`。不要按普通直觉改回 replay 前更新，否则吞吐测试会复现 `graph_task_update_end` 507000。

4. 当前只热更新 paged attention 的 `seq_lens/context_lens` 路径。若后续引入 spec decode、chunked prefill、prefix cache 特殊路径，可能还需要更新 block table 或其他动态图参数。

5. 当前默认 capture sizes 固定为 `[1, 2, 4, 8]`，这是为了先保证稳定和显存可控。吞吐优化阶段可以逐步扩大，例如 `[1, 2, 4, 8, 16]`，但每次都需要验证显存、capture 时间和输出正确性。

6. `FULL` 和 `PIECEWISE` 不应直接打开。若后续要支持，需要分别处理 prefill/mixed graph metadata，不应复用本轮 decode-only 逻辑。

7. `ASCEND_LAUNCH_BLOCKING=1` 会阻断 graph capture，当前实现会直接报错。这是预期行为。

## 14. 后续工程师快速验证步骤

1. 确认环境：

```bash
conda activate flagos
python - <<'PY'
import torch
import torch_npu
print(torch.npu.is_available())
print(torch.npu.get_device_name(0))
PY
```

2. 运行端到端验证：

```bash
VLLM_PLUGINS=fl /root/miniconda3/envs/flagos/bin/python test_vllm.py
```

3. 检查日志必须包含：

```text
Ascend NPU FULL_DECODE_ONLY graph enabled
cudagraph_capture_sizes=[1, 2, 4, 8]
Capturing CUDA graphs (decode, FULL): 4/4
```

4. 检查输出不要退化为重复标点、重复中文 token 或明显偏移。当前已验证的正确输出是：

```text
Prompt: 'Hello, my name is', Generated text: " Xiaoyu, and I'm a student at"
```

5. 若要调大 capture sizes，优先只改：

```python
ASCEND_DEFAULT_CUDAGRAPH_CAPTURE_SIZES = [1, 2, 4, 8]
```

并重新跑 `test_vllm.py` 做正确性验证，再做吞吐测试。

## 15. 追加排查：长上下文 PA workspace 不足导致 507000

本次重新查看 `benchmarks/flagos_eval/output.log` 后，确认 `chat_1k`、`chat_4k`、`chat_6k` 初始化阶段都已经启用了：

```text
cudagraph_mode=<CUDAGraphMode.FULL_DECODE_ONLY: (2, 0)>
cudagraph_capture_sizes=[1, 2, 4, 8]
```

因此问题不是 `chat_1k` / `chat_4k` 没有进入 FULL_DECODE_ONLY，而是 `chat_6k` 的长上下文更容易暴露 `_npu_paged_attention` graph update 的 workspace 问题。

崩溃点仍是：

```text
RuntimeError: graph_task_update_end ... error code is 507000
total_num_scheduled_tokens=2
num_scheduled_tokens={298: 1, 299: 1}
```

这说明崩溃发生在 decode-only graph replay/update 阶段，命中 capture size `2`。当时两个请求的上下文长度已经接近 `6k/7k`，而插件原先在 dummy graph capture 时沿用上游 CUDA 逻辑：

```python
seq_lens = max_query_len  # uniform decode capture 时通常为 1
```

`torch_npu._npu_paged_attention_get_workspace()` 返回的 workspace 与 capture 时看到的 `seq_lens` 有关。若 capture 时 `seq_lens=1`，后续 replay 时再用 task update 把 `context_lens` 热更新到几千 token，workspace 可能不足，Ascend runtime 会在 `graph_task_update_end()` 返回 `507000`。

参考 `/root/vllm-ascend/vllm_ascend/worker/model_runner_v1.py` 的处理，已在 `vllm_fl/worker/model_runner.py` 中补齐：

```python
ASCEND_SEQ_LEN_WITH_MAX_PA_WORKSPACE = 6144
```

并在 Ascend NPU、`FULL_DECODE_ONLY`、full decode graph capture 时，把 dummy `seq_lens` 提升到 `6144`：

```python
if (
    current_device_type == "npu"
    and is_graph_capturing
    and cudagraph_runtime_mode == CUDAGraphMode.FULL
    and configured_cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
):
    return ASCEND_SEQ_LEN_WITH_MAX_PA_WORKSPACE
```

普通非 capture 路径仍保持 `seq_lens=max_query_len`，mixed dummy batch 路径仍保持原来的短 dummy layout。

新增测试：

```text
tests/unit_tests/worker/test_model_runner.py::TestDummySeqLensForAttention
```

覆盖点：

- Ascend full decode graph capture 使用 `6144` 申请 PA workspace。
- 非 capture decode 仍保持 `max_query_len=1`。
- mixed dummy batch 保持原布局。

已通过：

```bash
/root/miniconda3/envs/flagos/bin/python -m py_compile \
  vllm_fl/worker/model_runner.py \
  tests/unit_tests/worker/test_model_runner.py
```

```bash
git diff --check -- \
  vllm_fl/worker/model_runner.py \
  tests/unit_tests/worker/test_model_runner.py
```

`flagos` 环境仍未安装 pytest，因此未在该环境运行新增 pytest：

```text
/root/miniconda3/envs/flagos/bin/python: No module named pytest
```

后续需要重新跑完整 `run_benchmark.sh`，重点确认 `chat_6k` 能否越过之前接近完成时的 `batch=2` 尾部崩溃点。
