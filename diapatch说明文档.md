# vllm_fl/dispatch 系统说明文档

## 📑 目录

1. [概述](#概述)
2. [核心架构](#核心架构)
3. [目录结构](#目录结构)
4. [核心模块解析](#核心模块解析)
5. [运行流程](#运行流程)
6. [配置系统](#配置系统)
7. [支持的算子](#支持的算子)
8. [使用示例](#使用示例)

---

## 概述

`vllm_fl/dispatch` 是一个**灵活的运算符分派系统**，它的核心功能是根据硬件环境、用户配置和可用性自动选择最优的算子实现。

### 核心特性

✅ **多后端支持**：FlagGems、厂商优化、PyTorch 参考实现  
✅ **动态选择**：根据硬件自动选择最优实现  
✅ **灵活配置**：支持 YAML 配置、环境变量、代码上下文  
✅ **线程安全**：内部使用锁保护共享资源  
✅ **插件扩展**：支持外部插件动态注册新的实现  
✅ **调试工具**：IO 转储、日志记录、性能分析  

---

## 核心架构

### 整体设计图

```
┌─────────────────────────────────────────────────────┐
│               用户代码                               │
│         call_op("silu_and_mul", x)                  │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│         OpManager (调度管理器)                       │
│  ┌──────────────────────────────────────────────┐  │
│  │ 1. 检查缓存                                   │  │
│  │ 2. 获取策略 (环境变量/上下文)                │  │
│  │ 3. 查询注册表获取所有实现                    │  │
│  │ 4. 按供应商白名单/黑名单过滤                │  │
│  │ 5. 检查可用性 (is_available())              │  │
│  │ 6. 按优先级和选择顺序排序                    │  │
│  │ 7. 缓存并返回选中的实现                      │  │
│  └──────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│         OpRegistry (运算符注册表)                   │
│  ┌──────────────┐ ┌──────────────┐ ┌───────────┐  │
│  │  FlagGems    │ │   Vendor     │ │Reference  │  │
│  │ Priority:150 │ │Priority:100  │ │Priority:50│  │
│  └──────────────┘ └──────────────┘ └───────────┘  │
└─────────────────────────────────────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │   具体实现函数       │
          │  (cuda/ascend/...)  │
          └──────────────────────┘
```

### 优先级流程

```
优先级排序 (从高到低)：

1️⃣  FlagGems (DEFAULT)   - 优先级 150
    │ ├─ 使用 FlagGems 库优化算子
    │ ├─ 需要 flag_gems 库可用
    │ └─ 最高优化程度
    │
2️⃣  Vendor 优化         - 优先级 100
    │ ├─ CUDA (NVIDIA)
    │ ├─ Ascend (华为)
    │ ├─ MetaX (麒麟)
    │ └─ 针对特定硬件优化
    │
3️⃣  PyTorch 参考        - 优先级 50
    └─ 使用 torch 原生实现
      └─ 最大兼容性，性能最低
```

---

## 目录结构

```
vllm_fl/dispatch/
├── __init__.py                  # 模块入口，导出公开 API
├── types.py                     # 核心类型定义 (OpImpl, BackendImplKind)
├── registry.py                  # 线程安全的运算符注册表
├── policy.py                    # 选择策略管理
├── manager.py                   # 核心调度管理器 (OpManager)
├── ops.py                       # 后端基类 (VLLMFLBackendBase)
├── builtin_ops.py              # 内置运算符注册
├── discovery.py                 # 插件发现机制
├── logger_manager.py            # 日志管理
├── io_common.py                 # IO 通用工具
├── io_dumper.py                 # IO 转储调试工具
├── README.md                    # 官方文档
│
├── config/                      # 平台特定配置
│   ├── __init__.py             # 配置加载器
│   ├── ascend.yaml             # Ascend NPU 默认配置
│   └── nvidia.yaml               # CUDA 默认配置
│
└── backends/                    # 后端实现
    ├── base.py                 # 后端抽象基类
    ├── flaggems/               # FlagGems 后端 (DEFAULT, 优先级 150)
    │   ├── flaggems.py         # 后端类
    │   ├── register_ops.py      # 注册函数
    │   └── impl/               # 算子实现
    │       ├── activation.py    # 激活函数 (silu_and_mul, gelu_and_mul)
    │       ├── normalization.py # 归一化 (rms_norm)
    │       ├── rotary.py        # 旋转位置编码 (rotary_embedding)
    │       ├── attention.py      # 实现了一个基于 FlashAttention 的 vLLM 注意力后端（AttentionFLBackend）
    │       └── custom_attention.py  # 注意力后端注册
    ├── reference/              # 参考后端 (PyTorch, 优先级 50)
    └── vendor/                 # 厂商特定后端 (优先级 100)
        ├── cuda/               # NVIDIA CUDA 后端
        │   └── impl/
        │       ├── activation.py
        │       ├── normalization.py
        │       └── rotary.py
        └── ascend/             # 华为 Ascend NPU 后端
            └── impl/
                ├── activation.py
                ├── normalization.py
                ├── rotary.py
                ├── attention.py     # AscendAttentionBackend
                └── attention_mask.py
```

---

## 核心模块解析

### 1. **类型定义** (`types.py`)

#### 枚举：`BackendImplKind`

```python
class BackendImplKind(str, Enum):
    DEFAULT = "flagos"      # FlagGems 优化实现
    REFERENCE = "reference" # PyTorch 参考实现
    VENDOR = "vendor"       # 厂商优化实现
```

#### 优先级常数：`BackendPriority`

```python
class BackendPriority:
    DEFAULT = 150    # FlagGems - 最优化
    VENDOR = 100     # 厂商优化 - 次优
    REFERENCE = 50   # PyTorch - 基准
```

#### 数据类：`OpImpl`

```python
@dataclass(frozen=True)
class OpImpl:
    """运算符实现描述符"""
    op_name: str           # 算子名 ("silu_and_mul", "rms_norm" 等)
    impl_id: str           # 唯一实现 ID ("default.flagos", "vendor.cuda" 等)
    kind: BackendImplKind  # 实现类型
    fn: Callable           # 实现函数
    vendor: Optional[str]  # 厂商名称 (VENDOR 类型时必须)
    priority: int          # 选择优先级 (值越高越优先)
    supported_dtypes: Optional[Set[str]]  # 支持的数据类型
    min_arch: Optional[str]               # 最低架构要求
    
    def is_available(self) -> bool:
        """检查实现在当前环境是否可用"""
```

#### 令牌匹配：`match_token()`

```python
# 支持的令牌格式
"flagos"           # 任何 DEFAULT 实现
"reference"        # 任何 REFERENCE 实现
"vendor"           # 任何 VENDOR 实现
"vendor:cuda"      # 特定厂商 CUDA
"vendor:ascend"    # 特定厂商 Ascend
"impl:default.flagos"  # 特定实现 ID
```

### 2. **注册表** (`registry.py`)

#### 线程安全的注册表

```python
class OpRegistry:
    """存储和查询运算符实现的线程安全注册表"""
    
    def register_impl(self, impl: OpImpl) -> None:
        """注册单个实现"""
    
    def register_many(self, impls: Sequence[OpImpl]) -> None:
        """注册多个实现"""
    
    def get_implementations(self, op_name: str) -> List[OpImpl]:
        """获取某个算子的所有实现"""
    
    def get_implementation(self, op_name: str, impl_id: str) -> Optional[OpImpl]:
        """获取特定实现"""
    
    def snapshot(self) -> OpRegistrySnapshot:
        """获取注册表快照 (用于选择策略)"""
```

**使用示例**：

```python
registry = OpRegistry()

# 注册实现
registry.register_impl(OpImpl(
    op_name="silu_and_mul",
    impl_id="default.flagos",
    kind=BackendImplKind.DEFAULT,
    fn=flaggems_silu_and_mul,
    priority=150
))

# 查询实现
impls = registry.get_implementations("silu_and_mul")
for impl in impls:
    print(f"{impl.impl_id} - priority: {impl.priority}")
```

### 3. **后端基类** (`ops.py`)

```python
class VLLMFLBackendBase(ABC):
    """所有后端必须继承的基类"""
    
    @abstractmethod
    def is_available(self) -> bool:
        """当前环境是否支持此后端"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """后端名称"""
    
    # ====== 激活函数 ======
    @abstractmethod
    def silu_and_mul(self, x: torch.Tensor) -> torch.Tensor:
        """SiLU + 乘法 (用于 Qwen3)"""
    
    @abstractmethod
    def gelu_and_mul(self, x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
        """GELU + 乘法 (用于 MoE)"""
    
    # ====== 归一化 ======
    @abstractmethod
    def rms_norm(self, x, residual, weight, epsilon) -> torch.Tensor:
        """RMS 归一化"""
    
    # ====== 位置编码 ======
    @abstractmethod
    def rotary_embedding(self, query, key, cos, sin, position_ids, 
                        rotary_interleaved=False, inplace=True):
        """旋转位置编码 (Qwen3 使用)"""
    
    # ====== 注意力 ======
    @abstractmethod
    def attention_backend(self, use_mla: bool = False, 
                         use_sparse: bool = False) -> str:
        """返回注意力后端实现的类路径"""
```

### 4. **调度管理器** (`manager.py`)

这是核心的运算符分派逻辑。

```python
class OpManager:
    """核心的运算符调度和缓存管理器"""
    
    def __init__(self, registry: OpRegistry):
        self.registry = registry
        self._cache = {}  # 缓存已选中的实现
    
    def resolve(self, op_name: str, 
                policy: Optional[SelectionPolicy] = None) -> Callable:
        """
        解析运算符 - 返回最适合的实现函数
        
        流程：
        1. 检查缓存
        2. 获取策略
        3. 从注册表获取所有实现
        4. 过滤 (厂商黑名单/白名单)
        5. 检查可用性
        6. 排序 (按优先级和选择顺序)
        7. 缓存并返回
        """
    
    def call(self, op_name: str, *args, **kwargs):
        """直接调用运算符"""
        fn = self.resolve(op_name)
        return fn(*args, **kwargs)
```

**解析流程详解**：

```python
# 假设有以下实现注册
FlagGems   - priority 150, available=True
CUDA       - priority 100, available=True
Reference  - priority 50, available=True

# 调用 resolve("silu_and_mul")
# ↓ 按优先级排序
# [FlagGems(150), CUDA(100), Reference(50)]
# ↓ 检查 is_available() 和黑名单/白名单
# [FlagGems ✓, CUDA ✓, Reference ✓]
# ↓ 返回第一个可用的
# → FlagGems 实现
```

### 5. **选择策略** (`policy.py`)

策略定义了如何选择实现。

```python
@dataclass
class SelectionPolicy:
    """运算符实现选择策略"""
    prefer: str = "flagos"  # 首选后端: "flagos"/"vendor"/"reference"
    strict: bool = False    # 严格模式: 失败时是否立即中止
    per_op_order: Dict[str, List[str]] = field(default_factory=dict)  # 每个算子的选择顺序
    allow_vendors: Optional[List[str]] = None  # 供应商白名单
    deny_vendors: Optional[List[str]] = None   # 供应商黑名单
```

**策略上下文**：

```python
from vllm_fl.dispatch import with_preference, with_strict_mode

# 临时使用优先选择 vendor 实现
with with_preference("vendor"):
    result = call_op("silu_and_mul", x)
    # 在此上下文中，vendor 实现被优先选择

# 临时启用严格模式
with with_strict_mode(True):
    result = call_op("rms_norm", x, residual, weight, epsilon)
    # 如果 rms_norm 失败，立即抛出异常，不尝试其他实现
```

---

## 运行流程

### 完整的调用流程

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 用户代码调用                                              │
│    result = call_op("silu_and_mul", x)                      │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 2. OpManager.call()                                         │
│    ├─ 调用 resolve("silu_and_mul")                         │
│    └─ 调用返回的函数 fn(x)                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 3. OpManager.resolve()                                      │
│    ├─ 检查缓存是否有"silu_and_mul"                         │
│    │  ├─ 若有，返回缓存的函数                              │
│    │  └─ 若无，继续下一步                                  │
│    │                                                        │
│    ├─ 获取当前策略                                          │
│    │  ├─ 优先级别：get_policy()                           │
│    │  ├─ 环境变量优先                                      │
│    │  └─ 代码上下文次之                                    │
│    │                                                        │
│    ├─ 查询注册表                                            │
│    │  └─ 获取"silu_and_mul"的所有实现                     │
│    │     [FlagGems, CUDA Vendor, PyTorch Reference]       │
│    │                                                        │
│    ├─ 过滤实现                                              │
│    │  ├─ 应用 allow_vendors 白名单                         │
│    │  ├─ 应用 deny_vendors 黑名单                          │
│    │  └─ 检查 is_available()                               │
│    │                                                        │
│    ├─ 排序实现                                              │
│    │  ├─ 按 priority 降序排列                              │
│    │  ├─ 考虑 per_op_order 自定义顺序                      │
│    │  └─ 优先级：flagos(150) > vendor(100) > reference(50) │
│    │                                                        │
│    ├─ 选择第一个可用                                        │
│    │  └─ FlagGems (priority 150, available)                │
│    │                                                        │
│    └─ 缓存结果                                              │
│       └─ _cache["silu_and_mul"] = fn                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
              ┌──────────────┐
              │  返回函数    │
              │  FlagGems 的 │
              │ silu_and_mul │
              └──────┬───────┘
                     │
                     ▼
        ┌────────────────────────┐
        │ 4. 执行具体实现        │
        │                        │
        │ result = fn(x)         │
        │        = silu_and_mul( │
        │          torch.Tensor) │
        └────────────────────────┘
```

---

## 配置系统

### 配置优先级

```
┌���────────────────────────────────────────────────────┐
│          配置优先级 (从高到低)                       │
├─────────────────────────────────────────────────────┤
│ 1. VLLM_FL_CONFIG  │ 用户配置文件 (完全覆盖)       │
│ 2. 环境变量         │ 覆盖特定项                    │
│ 3. 平台配置文件    │ ascend.yaml / nvidia.yaml       │
│ 4. 内置默认值      │ 代码中定义的默认值            │
└─────────────────────────────────────────────────────┘
```

### 环境变量速查表

#### 核心配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VLLM_FL_PREFER_ENABLED` | `true` | 全局开关 |
| `VLLM_FL_PREFER` | `flagos` | 首选后端: flagos/vendor/reference |
| `VLLM_FL_STRICT` | `0` | 严格模式: 1=失败即中止, 0=尝试备选 |
| `VLLM_FL_PLATFORM` | (自动) | 强制平台: ascend/cuda |

#### 白名单/黑名单

| 变量 | 说明 |
|------|------|
| `VLLM_FL_ALLOW_VENDORS` | 供应商白名单 |
| `VLLM_FL_DENY_VENDORS` | 供应商黑名单 |
| `VLLM_FL_FLAGOS_WHITELIST` | FlagGems 算子白名单 |
| `VLLM_FL_FLAGOS_BLACKLIST` | FlagGems 算子黑名单 |
| `VLLM_FL_OOT_WHITELIST` | OOT 算子白名单 |
| `VLLM_FL_OOT_BLACKLIST` | OOT 算子黑名单 |

#### 高级配置

| 变量 | 说明 |
|------|------|
| `VLLM_FL_PER_OP` | 每个算子的选择顺序: `op1=a\|b\|c;op2=x\|y` |
| `VLLM_FL_LOG_LEVEL` | 日志级别: DEBUG/INFO/WARNING/ERROR |
| `VLLM_FL_DISPATCH_DEBUG` | 启用调度调试: 1/0 |

### YAML 配置示例

```yaml
# vllm_fl_dispatch.yaml

# 首选后端
prefer: vendor

# 严格模式 (失败立即中止，不尝试备选)
strict: false

# 供应商白名单
allow_vendors:
  - cuda
  - ascend

# 供应商黑名单 (与白名单互斥)
# deny_vendors:
#   - iluvatar

# 每个算子的后端选择顺序
op_backends:
  # 对 rms_norm: 优先 vendor，再 flagos，最后 reference
  rms_norm:
    - vendor
    - flagos
    - reference
  
  # 对 silu_and_mul: 仅尝试 CUDA vendor 和 flagos，不尝试 reference
  silu_and_mul:
    - vendor:cuda
    - flagos
  
  # 对 rotary_embedding: vendor:ascend > flagos > reference
  rotary_embedding:
    - vendor:ascend
    - flagos
    - reference

# FlagGems 算子黑名单
flagos_blacklist:
  - mm
  - to_copy
  - zeros

# OOT 算子黑名单
oot_blacklist:
  - fused_moe
  - unquantized_fused_moe_method
```

---

## 支持的算子

### Qwen3 相关算子

| 算子 | 说明 | Qwen3 用途 |
|------|------|-----------|
| `silu_and_mul` | SiLU 激活 + 乘法 | 前馈网络中的门控激活函数 |
| `rms_norm` | RMS 层归一化 | 每层的输入/输出归一化 |
| `rotary_embedding` | 旋转位置编码 | 位置信息编码 |
| `attention_backend` | 注意力后端选择 | 核心注意力计算 |

### 三层优化实现

```
silu_and_mul:
  ├─ FlagGems      (priority 150) → flaggems.impl.activation
  ├─ CUDA Vendor   (priority 100) → vendor.cuda.impl.activation
  ├─ Ascend Vendor (priority 100) → vendor.ascend.impl.activation
  └─ PyTorch Ref   (priority 50)  → torch.nn.functional.silu

rms_norm:
  ├─ FlagGems      (priority 150) → flaggems.impl.normalization
  ├─ CUDA Vendor   (priority 100) → vendor.cuda.impl.normalization
  ├─ Ascend Vendor (priority 100) → vendor.ascend.impl.normalization
  └─ PyTorch Ref   (priority 50)  → torch.nn.RMSNorm

rotary_embedding:
  ├─ FlagGems      (priority 150) → flaggems.impl.rotary
  ├─ CUDA Vendor   (priority 100) → vendor.cuda.impl.rotary
  ├─ Ascend Vendor (priority 100) → vendor.ascend.impl.rotary
  └─ PyTorch Ref   (priority 50)  → 自定义实现

attention_backend:
  ├─ FlagGems      → vllm_fl.dispatch.backends.flaggems.impl.attention
  ├─ CUDA Vendor   → vllm_fl.dispatch.backends.vendor.cuda.impl.attention
  └─ Ascend Vendor → vllm_fl.dispatch.backends.vendor.ascend.impl.attention
```

---

## 使用示例

### 基础用法

```python
from vllm_fl.dispatch import call_op, resolve_op

# 方法 1: 直接调用
result = call_op("silu_and_mul", x)

# 方法 2: 先解析再调用
fn = resolve_op("rms_norm")
result = fn(x, residual, weight, epsilon=1e-6)
```

### 使用管理器

```python
from vllm_fl.dispatch import get_default_manager

manager = get_default_manager()

# 解析操作
fn = manager.resolve("rotary_embedding")
q, k = fn(query, key, cos, sin, position_ids)

# 直接调用
result = manager.call("silu_and_mul", x)
```

### 代码级策略控制

```python
from vllm_fl.dispatch import (
    with_preference,
    with_strict_mode,
    with_allowed_vendors,
    call_op
)

# 临时使用 vendor 实现
with with_preference("vendor"):
    result = call_op("rms_norm", x, residual, weight, eps)

# 启用严格模式 (失败立即中止)
with with_strict_mode(True):
    result = call_op("silu_and_mul", x)

# 仅允许特定供应商
with with_allowed_vendors(["cuda"]):
    result = call_op("rotary_embedding", q, k, cos, sin, pos_ids)

# 嵌套上下文
with with_preference("flagos"):
    with with_strict_mode(False):
        result = call_op("silu_and_mul", x)
```

### Qwen3 推理完整示例

```python
import os
import torch
from vllm import LLM, SamplingParams
from vllm_fl.dispatch import call_op, with_preference

# 配置环境
os.environ["VLLM_FL_PREFER"] = "flagos"
os.environ["USE_FLAGGEMS"] = "1"

# 创建 LLM (自动注册 dispatch)
llm = LLM(
    model="Qwen/Qwen3-4B",
    tensor_parallel_size=1,
    max_model_len=8192,
)

# 推理
prompts = ["Hello, how are you?"]
sampling_params = SamplingParams(max_tokens=100, temperature=0.7)
outputs = llm.generate(prompts, sampling_params)

# 在推理过程中，dispatch 会自动选择最优实现：
# - silu_and_mul → FlagGems (如果可用) 
# - rms_norm     → FlagGems (如果可用)
# - rotary_embedding → FlagGems (如果可用)
# - attention_backend → 根据硬件选择

for output in outputs:
    print(output.outputs[0].text)
```

### 性能调优示例

```python
import os
from vllm_fl.dispatch import call_op

# 场景 1: GPU 优化 (NVIDIA)
os.environ["VLLM_FL_PREFER"] = "vendor"
os.environ["VLLM_FL_ALLOW_VENDORS"] = "cuda"
# 直接使用 CUDA 优化实现

# 场景 2: CPU 兼容性优先
os.environ["VLLM_FL_PREFER"] = "reference"
# 使用 PyTorch 原生实现，最大兼容性

# 场景 3: FlagGems 仅用于特定算子
os.environ["VLLM_FL_FLAGOS_WHITELIST"] = "silu_and_mul,rms_norm"
# 只有这两个算子使用 FlagGems，其他使用 vendor/reference

# 场景 4: 禁用特定实现
os.environ["VLLM_FL_FLAGOS_BLACKLIST"] = "mm,to_copy"
# FlagGems 不用于 mm 和 to_copy 算子

# 场景 5: 严格模式调试
os.environ["VLLM_FL_STRICT"] = "1"
# 如果首选实现失败，立即抛出异常
```

### 调试和日志

```python
import os
from vllm_fl.dispatch import set_log_level, get_logger, call_op

# 启用调试日志
set_log_level("DEBUG")

# 启用调度调试输出
os.environ["VLLM_FL_DISPATCH_DEBUG"] = "1"

# 获取日志记录器
logger = get_logger("vllm_fl.dispatch")
logger.debug("Dispatch debug info")

# 调用操作 (会输出详细调试信息)
result = call_op("silu_and_mul", x)
```

---

## 架构优点

| 优点 | 说明 |
|------|------|
| 🎯 **自适应** | 根据硬件自动选择最优实现 |
| 🔌 **可扩展** | 支持插件添加新的后端实现 |
| ⚡ **高效** | 实现缓存避免重复选择 |
| 🔒 **线程安全** | 内部使用锁保护共享资源 |
| 🛠️ **可调试** | 支持日志、IO 转储、调试模式 |
| 📋 **灵活配置** | YAML、环境变量、代码上下文三层配置 |
| 🎓 **易学易用** | 简洁的 API: `call_op()`, `resolve_op()` |

---

## 总结

`vllm_fl/dispatch` 系统是一个**智能的运算符分派引擎**，它：

1. **自动选择**：根据硬件和可用性自动选择最优实现
2. **灵活配置**：支持多种配置方式满足不同场景
3. **高性能**：通过缓存和多层优化加速算子执行
4. **易于扩展**：支持插件添加新的后端实现

对于 **Qwen3 模型**，dispatch 系统会自动优化关键算子：
- `silu_and_mul` - Qwen3 激活函数
- `rms_norm` - 层归一化
- `rotary_embedding` - 位置编码
- `attention_backend` - 注意力计算（包括 MLA 支持）

通过正确配置和使用 dispatch 系统，可以在保证兼容性的同时获得最优的推理性能。