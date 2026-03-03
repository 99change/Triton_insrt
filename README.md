# Triton PTX Instrumentation

对 Triton kernel 编译产生的 PTX 进行自动插桩，在每个基本块的入口和出口插入时钟采样探针，从而获得 GPU 上各 sub-block 的执行耗时。

---

## 系统组成

```
Triton_insrt/
├── ominiscope/                 # 核心组件
│   ├── config.ini              # 插桩配置文件
│   ├── probe/                  # PTX 探针模板
│   │   ├── head.ptx            # 寄存器声明 / 初始化代码段
│   │   ├── entry.ptx           # 探针入口模板（START：记录起始时间戳，INDEX 编码在 clock_hi 高 16 位）
│   │   └── exit.ptx            # 探针出口模板（END：记录结束时间戳并写入 profiler_buffer）
│   └── python/                 # Python 实现
│       ├── __init__.py         # 对外暴露 ominiscopeManager
│       ├── manager.py          # 插桩引擎 + 结果收集
│       ├── insert_ptx.py       # PTX 插桩逻辑
│       ├── subBlock.py         # PTX 解析器：把 PTX 切分成 sub_block
│       └── sanitizer.py        # 探针消除与合并
│
├── examples/
│   ├── matmul.py               # 端到端示例：matmul kernel 完整插桩流程
│   └── with_proton.py          # 结合 Proton 使用的示例
│
├── analysis/                   # 运行过程中分析插桩数据生成的配置文件
│   ├── map.json                # 探针序号与源代码行号的映射
│   └── active.json             # 活跃的插针列表（经 sanitizer 处理后）
│
└── output/                     # 运行时自动生成
    ├── original.ptx            # Triton 编译出的原始 PTX（插桩前）
    ├── instrumented.ptx       # 插桩后的 PTX
    ├── raw.csv                 # 原始时间戳数据（每线程每探针）
    └── trace.json              # Chrome trace 格式的可视化探针数据
```

---

## 各组件说明

### `ominiscope/python/subBlock.py`

把 PTX 文本解析成一组 `subBlock` 对象。

**`subBlock` 的字段：**

| 字段 | 含义 |
|---|---|
| `entry` / `exit` | 该 block 在 PTX 文件中的起止行号（1-based） |
| `file` | `.loc <file> <line> <col>` 中的源文件索引 |
| `loc` | `.loc` 指令中的源代码行号 |
| `mma_lines` | 含 `mma` 指令的行号列表（1-based） |
| `empty_lines` | 空行及注释行的行号列表（1-based） |
| `is_start` | 是否为 kernel 入口 `{` |
| `is_end` | 是否为 `ret;` |
| `is_sync` | 是否含有 `mbarrier.try_wait.parity.acquire`（cuTile 异步等待块） |

**切块规则：**
- 遇到 `$L__BB` 标签 → 开启新 block
- 遇到无条件/条件跳转 → 关闭当前 block
- `bar.sync` 不触发切块，视为普通指令（Triton 特有行为，与 cuTile 不同）
- 只有 `mbarrier.try_wait.parity.acquire` 才会触发 `is_sync`，`bar.sync` 按普通指令处理

**`.loc` 指令说明：**

Triton 编译产生的 PTX 中，`.loc` 指令格式为 `.loc <文件索引> <行号> <列号>`，其中文件索引标识该指令来自哪个 Python 源文件（一个 kernel 可能引用多个源文件，例如 kernel 本体与 Triton 标准库）。`builder()` 会将最近一条 `.loc` 的 `file` 和 `loc` 赋给后续 block，直到下一条 `.loc` 出现。

**`builder(lines)` 函数：**

| 参数 | 说明 |
|---|---|
| `lines` | PTX 文本按行组成的列表（含或不含行尾 `\n` 均可） |

返回值：按源文件出现顺序排列的 `subBlock` 对象列表。

---

### `ominiscope/python/manager.py`

核心插桩引擎，对外暴露 `ominiscopeManager` 类。

**工作流程：**

1. 创建 `ominiscopeManager()` 实例，自动 monkey-patch `CUDABackend.make_cubin` 和 `JITFunction.run`
2. 用户正常调用 Triton kernel，触发编译
3. 编译时 `make_cubin` 被拦截，原始 PTX 被截获，保存为 `output/original.ptx`
4. 调用 `insert_ptx.instrument_ptx()` 对 PTX 进行插桩：
   - 用 `builder()` 把 PTX 切分成 subBlock 列表
   - 过滤掉 `file != target_file` 的 block（排除非目标源文件）
   - 在每对 (block_entry, block_exit) 插入 START/END 探针，分配 INDEX 编号（0-based）
   - 在 PTX 头部插入寄存器声明（来自 `probe/head.ptx`）
5. 插桩后 PTX 保存为 `output/instrumented.ptx`，送回 ptxas 编译成 cubin
6. Kernel 运行结束后，调用 `export_raw()` 从 GPU buffer 读回时间戳并计算耗时
7. 调用 `export_trace()` 导出 Chrome Trace 格式

#### `ominiscopeManager` 构造参数

从 `ominiscope/config.ini` 读取配置，也可手动传入参数覆盖：

| 参数 | 含义 | 默认值 |
|---|---|---|
| `mode` | 插桩模式：`block`（基本块）、`single`（每条指令）、`entire`（整个 kernel） | 来自 config |
| `target_file` | 目标源文件索引（来自 `.loc` 指令）；初次运行可从日志中找到正确值 | 1 |
| `target_kernel` | 目标 kernel 名称；`None` 表示对所有 kernel 插桩 | 来自 config |
| `t_start` / `t_end` | 采样线程的全局 thread ID 范围 | 0 / 127 |
| `l_buffer` | buffer slot 数量（每个探针触发占用一个 slot） | 4096 |
| `output_dir` | 原始采样输出目录（PTX、时间戳、Trace） | "output" |
| `analysis_dir` | 分析元数据目录（`map.json`） | "analysis" |

#### `config.ini` 各字段说明：

`[ominiscope]` 节：

| 字段 | 含义 | 默认值 |
|---|---|---|
| `target_kernel` | 目标 kernel 名称（设为空则对所有 kernel 插桩） | `matmul_kernel` |
| `target_file` | 目标源文件索引（来自 `.loc` 指令） | `1` |
| `mode` | 插桩模式 | `block` |
| `t_start` | 采样线程全局 ID 下界（含） | `0` |
| `t_end` | 采样线程全局 ID 上界（含） | `127` |
| `l_buffer` | buffer slot 数量 | `4096` |

`[probe]` 节：

| 字段 | 含义 |
|---|---|
| `probe_dir` | 探针模板目录 |
| `head` / `entry` / `exit` | 各探针模板文件名 |

`[analysis]` 节：

| 字段 | 含义 | 默认值 |
|---|---|---|
| `analysis_dir` | 分析元数据目录 | `analysis` |
| `map` | 探针→源码行映射 JSON 文件名 | `map.json` |
| `active` | 有效探针列表输出 JSON 文件名 | `active.json` |

`[output]` 节：

| 字段 | 含义 | 默认值 |
|---|---|---|
| `output_dir` | 原始采样输出目录 | `output` |
| `raw_base` | 原始时间戳文件名前缀 | `raw` |
| `trace` | Chrome Trace JSON 文件名 | `trace.json` |
| `dump` | 是否保存原始和插桩后的 PTX 文件 | `True` |

`[elimination]` 节：

| 字段 | 含义 | 默认值 |
|---|---|---|
| `enable` | 是否启用探针消除 | `True` |
| `dead` | 开启死探针消除（过滤从未被执行的探针） | `True` |
| `merge` | 开启相邻探针合并（合并同源码行的 1:1 相邻探针对） | `True` |

#### `export_raw()` 方法

从 GPU buffer 解析时间戳数据并保存为文件。

**Buffer 内存布局：**

探针索引（INDEX = `probe_idx << 16`）编码在 `clock_hi` 的高 16 位，低 16 位保留真实时钟高位。exit 探针每次触发写入一行：

| 字段 | 含义 |
|---|---|
| `start_lo` | 入口 `%clock` 低 32 位 |
| `start_hi` | `(clock_hi & 0xFFFF) \| (probe_idx << 16)` |
| `end_lo` | 出口 `%clock` 低 32 位 |
| `end_hi` | `(clock_hi & 0xFFFF) \| (probe_idx << 16)` |

**解析流程：**
1. 将 `profiler_buffer` 以 u32 视图 reshape 为 `[n_slots, n_threads, 4]`
2. 裁掉末尾全零的 slot（未写入的 slot 值全为 0）
3. 对每个 slot 每个线程：
   - 重建 64 位时间戳：`time = ((hi & 0xFFFF) << 32) | lo`
   - 提取探针编号：`probe_idx = start_hi >> 16`
4. 保存为 `<output_dir>/raw.npy` 和 `<output_dir>/raw.csv`

#### `print_summary()` 方法

以表格形式打印各探针的 Probe ID、源码行号、平均/最小/最大时钟周期数。

#### `export_trace(path)` 方法

将探针数据导出为 Chrome Trace Event JSON 格式，可在 `chrome://tracing` 或 [Perfetto UI](https://ui.perfetto.dev) 中打开。

---

### `ominiscope/python/insert_ptx.py`

PTX 插桩核心逻辑，对外暴露 `instrument_ptx()` 函数。

```python
instrument_ptx(ptx_str, mode, t_start, t_end, target_file, active_list) -> (modified_ptx, loc_map, n_probes)
```

| 参数 | 含义 | 默认值 |
|---|---|---|
| `ptx_str` | Triton 编译产生的原始 PTX 字符串 | — |
| `mode` | 插桩模式：`block`/`single`/`entire` | `"block"` |
| `t_start` / `t_end` | 采样线程的全局 thread ID 范围 | `0` / `127` |
| `target_file` | 仅对 `.loc` 来自该文件索引的 block 插桩 | `1` |
| `active_list` | 有效探针列表 JSON 文件路径；传 `None` 禁用消除 | `None` |

返回值：
- `modified_ptx`：插桩后的 PTX 字符串
- `loc_map`：`{probe_id: Python 源码行号}` 的字典
- `n_probes`：插入的探针对数量

---

### `ominiscope/python/sanitizer.py`

独立的探针后处理模块，提供死探针消除和相邻探针合并功能。

**死探针消除原理：** 找出在所有线程中 `start_idx` / `end_idx` 至少出现过一次的探针索引，过滤掉从未被任何线程执行的探针。

**相邻探针合并原理：** 分析相邻探针对之间的 `end_idx → next start_idx` 转移关系，若某对探针满足"1:1 唯一连接"且来自同一 Python 源码行，则将其合并，降低插针粒度。

---

### PTX 探针模板（`ominiscope/probe/`）

来自 cuTile 的现成探针，插桩时填入 INDEX 占位符后直接拼接进 PTX。

- `head.ptx`：在 kernel 入口声明探针所需的额外寄存器，并通过参数绑定 `profiler_buffer`
- `entry.ptx`：START 探针，读取 `%clock` 和 `%clock_hi`，将 `INDEX`（= `probe_idx << 16`）编码到 `start_hi` 高 16 位，暂存于寄存器
- `exit.ptx`：END 探针，同上读取 end 时钟，将 `[start_lo, start_hi, end_lo, end_hi]` 以 `st.global.v4.u32` 原子写入 `profiler_buffer`，并推进写指针

---

## 使用方法

### 1. 导入 ominiscopeManager

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ominiscope', 'python'))
from manager import ominiscopeManager
```

### 2. 创建 Manager 实例并运行 Kernel

`ominiscopeManager` 是通用的，适用于任何 Triton kernel

```python
manager = ominiscopeManager()

# 正常运行 Triton kernel（会自动注入 profiler_buffer 参数）
c = torch.empty((M, N), device="cuda", dtype=torch.float16)
matmul_kernel[grid](
    a, b, c,
    M, N, K,
    ...
)
torch.cuda.synchronize()

# 导出结果
manager.export_raw()
manager.export_trace()
```

完整可运行示例见 `examples/matmul.py`，直接执行即可：

```bash python examples/matmul.py
```

### 3. 查看输出文件

**`output/` 目录（原始采样数据）：**

| 文件 | 内容 |
|---|---|
| `output/original.ptx` | 插桩前的原始 Triton PTX |
| `output/instrumented.ptx` | 插桩后的 PTX，可用于人工核查探针位置 |
| `output/raw.npy` | 原始时间戳二维数组（`int64`，形状 `[n_probes, n_threads × 4]`） |
| `output/raw.csv` | 同上内容的 CSV 格式（无表头） |
| `output/trace.json` | Chrome Trace Event JSON，可在 Perfetto UI 中可视化 |

**`analysis/` 目录（分析元数据）：**

| 文件 | 内容 |
|---|---|
| `analysis/map.json` | 探针 ID → Python 源码行号的映射（插桩时生成） |
| `analysis/active.json` | 经过死探针消除和合并后的有效探针列表（由 sanitizer 生成） |

---

## 配置说明

### 目标文件与 Kernel 识别

初次运行时，在 `config.ini` 中设置 `target_kernel` 为你要插桩的 kernel 名称，设置 `target_file` 为源文件索引（可在日志中看到 `.loc` 指令涉及的文件索引）。

### 插桩模式

| `mode` | 行为 |
|---|---|
| `block` | 在每个基本块的入口和出口各插入一个探针（标准用法） |
| `single` | 对每条 mma 指令单独计时；非 mma 指令也每条单独计时 |
| `entire` | 只在整个 kernel 的首尾各插一个探针，测量整体耗时 |

### Chrome Trace 可视化

运行后会在 `output/trace.json` 生成 Chrome Trace Event 格式的文件，可通过以下方式查看：

1. 打开 Chrome 浏览器，访问 `chrome://tracing`
2. 点击 "Load" 按钮，加载 `output/trace.json`

或访问 [Perfetto UI](https://ui.perfetto.dev)，通过 "Open trace file" 加载同一文件。
