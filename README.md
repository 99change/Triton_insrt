# Triton PTX Instrumentation

对 Triton kernel 编译产生的 PTX 进行自动插桩，在每个基本块的入口和出口插入时钟采样探针，从而获得 GPU 上各 sub-block 的执行耗时。

---

## 系统组成

```
Triton_insrt/
├── tri_ins/                   # 核心组件
│   ├── __init__.py            # 对外暴露 TritonInstrument
│   ├── ptx_parser.py          # PTX 解析器：把 PTX 切分成 sub_block
│   ├── config.ini             # 插桩配置脚本
│   ├── triton_instrument.py   # 插桩引擎 + 结果收集
│   ├── sanitizer.py           # 清理与融合插桩
│   └── probe
│       ├── head.ptx           # 寄存器声明 / 初始化代码段
│       ├── entry.ptx          # 探针入口模板（START：记录起始时间戳，INDEX 编码在 clock_hi 高 16 位）
│       └── exit.ptx           # 探针出口模板（END：记录结束时间戳并写入 profile_buffer）
│
├── examples/
│   └── test_instrument.py     # 端到端示例：matmul kernel 完整插桩流程
│
├── analysis/                  # 运行过程中分析插桩数据生成的配置文件
│   ├── active_probes.json     # 活跃的插桩列表
│   └── loc_map.json           # 插桩序号与源代码行号的映射
│
└── output/                    # 运行时自动生成
    ├── original.ptx           # Triton 编译出的原始 PTX（插桩前）
    ├── instrumented.ptx       # 插桩后的 PTX
    ├── raw.csv                # 原始时间戳数据（每线程每探针）
    └── trace.json             # Chrome trace 格式的可视化探针数据
```

---

## 各组件说明

### `tri_ins/ptx_parser.py`

把 PTX 文本解析成一组 `sub_block` 对象。

**`sub_block` 的字段：**

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

返回值：按源文件出现顺序排列的 `sub_block` 对象列表。

---

### `tri_ins/triton_instrument.py`

核心插桩引擎，对外暴露 `instrument_ptx()` 函数和 `TritonInstrument` 上下文管理器。

**工作流程：**

1. 用 `with TritonInstrument(...) as inst:` 上下文进入时，monkey-patch `CUDABackend.make_cubin`
2. 用户在 `with` 块内正常调用 Triton kernel，触发编译
3. 编译时 `make_cubin` 被拦截，原始 PTX 被截获，保存为 `output/original.ptx`
4. 调用 `instrument_ptx()` 对 PTX 进行插桩：
   - 用 `builder()` 把 PTX 切分成 sub_block 列表
   - 过滤掉 `file != target_file` 的 block（排除非目标源文件）
   - 在每对 (block_entry, block_exit) 插入 START/END 探针，分配 INDEX 编号（0-based）
   - 在 PTX 头部插入寄存器声明（来自 `head.ptx`）
5. 插桩后 PTX 保存为 `output/instrumented.ptx`，送回 ptxas 编译成 cubin
6. Kernel 运行结束后，调用 `get_results()` 从 GPU buffer 读回时间戳并计算耗时

#### `instrument_ptx()` 函数

```python
instrument_ptx(ptx_str, mode, t_start, t_end, target_file, active_list) -> (modified_ptx, loc_map, n_probes)
```

| 参数 | 含义 | 默认值 |
|---|---|---|
| `ptx_str` | Triton 编译产生的原始 PTX 字符串 | — |
| `mode` | 插桩模式，见下表 | `"block"` |
| `t_start` / `t_end` | 采样线程的全局 thread ID 范围 | `0` / `127` |
| `target_file` | 仅对 `.loc` 来自该文件索引的 block 插桩；第一次运行会打印所有遇到的文件索引，据此确定目标值 | `1` |
| `active_list` | 有效探针列表 JSON 文件路径（由 sanitizer 生成）；传 `None` 禁用消除，保留全部探针 | `None` |

返回值：
- `modified_ptx`：插桩后的 PTX 字符串
- `loc_map`：`{probe_id: Python 源码行号}` 的字典
- `n_probes`：插入的探针对数量

**插桩模式说明：**

| `mode` | 行为 |
|---|---|
| `"block"` | 在每个基本块的入口和出口各插入一个探针（标准用法） |
| `"single"` | 对每条 mma 指令单独计时；非 mma 指令也每条单独计时 |
| `"entire"` | 只在整个 kernel 的首尾各插一个探针，测量整体耗时 |
| `"config"` | 插入线程过滤配置探针（来自 `config.ptx`），无时间戳采样 |

#### `TritonInstrument` 构造参数

| 参数 | 含义 | 默认值 |
|---|---|---|
| `mode` | 插桩模式，同上 | `"block"` |
| `t_start` / `t_end` | 采样线程的全局 thread ID 范围 | `0` / `127` |
| `target_file` | 目标源文件索引 | `1` |
| `output_dir` | 原始采样输出目录（PTX、时间戳、Trace）；传 `None` 禁用 | `"output"` |
| `analysis_dir` | 分析元数据目录（`loc_map.json`）；传 `None` 禁用 | `"analysis"` |
| `active_list` | 有效探针列表 JSON 文件完整路径；传 `None` 禁用消除（由 `from_config` 根据 `elimination` 选项自动设置） | `None` |
| `n_probes_estimate` | 预估探针数量（供 `allocate_buffer` 默认使用） | `256` |
| `raw_csv` | 原始时间戳 CSV/NPY 文件名（不含扩展名后缀） | `"raw.csv"` |
| `trace_json` | Chrome Trace 输出文件名 | `"trace.json"` |
| `loc_map_json` | 探针→源码行映射 JSON 文件名 | `"loc_map.json"` |
| `dump_ptx` | 是否保存原始和插桩后的 PTX 文件 | `True` |

#### `from_config(config_path)` 类方法

从 INI 配置文件创建 `TritonInstrument` 实例。`config_path` 默认为 `tri_ins/config.ini`。

**`config.ini` 各字段说明：**

`[instrument]` 节：

| 字段 | 含义 | 默认值 |
|---|---|---|
| `mode` | 插桩模式，同 `TritonInstrument(mode=...)` | `block` |
| `target_file` | 目标源文件索引（来自 `.loc` 指令）；初次运行可从日志中的 "Source file indices seen in .loc" 找到正确值 | `1` |
| `t_start` | 采样线程全局 ID 下界（含） | `0` |
| `t_end` | 采样线程全局 ID 上界（含） | `127` |
| `n_probes_estimate` | 预估 buffer slot 数量，用于预分配；需考虑循环触发次数（同一静态探针在循环中每次触发占一个 slot），建议适当放大 | `1024` |

`[output]` 节：

| 字段 | 含义 | 默认值 |
|---|---|---|
| `output_dir` | 原始采样输出目录（相对于调用脚本的工作目录） | `output` |
| `raw_csv` | 原始时间戳 CSV 文件名 | `raw.csv` |
| `trace_json` | Chrome Trace JSON 文件名（可在 `chrome://tracing` 或 Perfetto 中查看） | `trace.json` |
| `dump_ptx` | 是否保存原始和插桩后的 PTX 文件 | `True` |

`[analysis]` 节：

| 字段 | 含义 | 默认值 |
|---|---|---|
| `analysis_dir` | 分析元数据目录（`loc_map.json`、`active_probes.json`） | `analysis` |
| `loc_map` | 探针→源码行映射 JSON 文件名 | `loc_map.json` |
| `active_list` | 有效探针列表输出 JSON 文件名（由 sanitizer 生成） | `active_probes.json` |

`[sanitizer]` 节：

| 字段 | 含义 | 默认值 |
|---|---|---|
| `enable` | 是否在 `from_config()` 工作流中启用探针消除（读取 `active_list` 并过滤无效探针）；独立 CLI 调用不受此影响 | `False` |
| `dead` | 开启死探针消除（过滤从未被执行的探针） | `True` |
| `merge` | 开启相邻探针合并（合并同源码行的 1:1 相邻探针对） | `True` |

#### `allocate_buffer(n_probes_estimate, n_threads)` 方法

预分配一个 GPU buffer，返回 `time_buffer`，需作为 kernel 最后一个 buffer 参数传入。

**Buffer 内存布局：**

探针索引（INDEX = `probe_idx << 16`）编码在 `clock_hi` 的高 16 位，低 16 位保留真实时钟高位。exit 探针每次触发写入一行：

| 字段 | 含义 |
|---|---|
| `start_lo` | 入口 `%clock` 低 32 位 |
| `start_hi` | `(clock_hi & 0xFFFF) \| (probe_idx << 16)` |
| `end_lo` | 出口 `%clock` 低 32 位 |
| `end_hi` | `(clock_hi & 0xFFFF) \| (probe_idx << 16)` |

| Buffer | 类型 | 大小 | 含义 |
|---|---|---|---|
| `time_buffer` | `int32`（存 u32） | `n_slots × n_threads × 4` | 每 slot 每线程：`[start_lo, start_hi, end_lo, end_hi]`，步长 = `n_threads × 16` 字节 |

#### `get_results(export_path=None)` 方法

从 GPU buffer 解析时间戳数据，同时可选地将结果保存为文件。

**解析流程：**
1. 将 `time_buffer` 以 u32 视图 reshape 为 `[n_slots, n_threads, 4]`
2. 裁掉末尾全零的 slot（未写入的 slot 值全为 0）
3. 对每个 slot 每个线程：
   - 重建 64 位时间戳：`time = ((hi & 0xFFFF) << 32) | lo`
   - 提取探针编号：`probe_idx = start_hi >> 16`
4. 按 `[probe_idx, probe_idx, start_time, end_time]` 顺序拼成一行，得到形状为 `[n_actual_slots, n_threads × 4]` 的二维 numpy 数组（`int64`）
5. 若提供 `export_path`（不含扩展名），则同时保存：
   - `<export_path>.npy`：原始二维数组（`int64`）
   - `<export_path>.csv`：同上内容的 CSV 格式（无表头）

返回 dict：

```python
{
    slot_id: {
        'probe_site':   int,    # 探针编号（= INDEX >> 16，对应 loc_map 的键）
        'loc':          int,    # Python 源码行号
        'start_idx':    Tensor, # 各线程的探针入口编号（int64）
        'end_idx':      Tensor, # 各线程的探针出口编号（int64）
        'start_time':   Tensor, # 入口 clock64 时间戳（int64，每线程）
        'end_time':     Tensor, # 出口 clock64 时间戳（int64，每线程）
        'duration':     Tensor, # end_time - start_time（int64，每线程）
        'mean_cycles':  float,  # duration 均值（clock cycles）
    }
}
```

#### `print_summary(results)` 方法

以表格形式打印各探针的 Probe ID、源码行号、平均/最小/最大时钟周期数。

#### `export_chrome_trace(path, results)` 方法

将探针数据导出为 Chrome Trace Event JSON 格式，可在 `chrome://tracing` 或 [Perfetto UI](https://ui.perfetto.dev) 中打开。

每个探针对应一个 `"X"`（complete）事件，字段含义：

| 字段 | 含义 |
|---|---|
| `ts` | 入口时间戳（clock cycles） |
| `dur` | 持续时间 = `end_time - start_time` |
| `tid` | 线程在采样范围内的相对索引 |
| `pid` | 固定为 `0` |
| `name` | `"Line <loc>"`（对应 Python 源码行） |

底层使用多线程 + 队列流水线写入，`chunk_size`（默认 4096）和 `workers`（默认 16）可调。

#### `sanitize(path, results, dead, merge)` 方法

对插桩结果进行后处理，生成有效探针列表并保存为 JSON。

| 参数 | 含义 | 默认值 |
|---|---|---|
| `dead` | 开启死探针消除（过滤掉从未被执行的探针） | `True` |
| `merge` | 开启相邻探针合并（将 1:1 连接且来自同一源码行的探针对合并） | `True` |

输出 JSON 格式：`{"active_start": {"list": [...]}, "active_end": {"list": [...]}}`

---

### `tri_ins/sanitizer.py`

独立的探针后处理模块，可作为命令行工具单独运行：

```bash
python -m tri_ins.sanitizer
```

从 `tri_ins/config.ini` 读取配置，从 `output/` 加载 `raw.csv`，从 `analysis/` 加载 `loc_map.json`，执行死探针消除和/或相邻探针合并，将有效探针列表写入 `analysis/active_probes.json`。

**`sanitize(raw_csv, loc_map_path, output_path, dead, merge)` 函数：**

| 参数 | 含义 |
|---|---|
| `raw_csv` | 原始时间戳 CSV 路径（行=探针，列=线程×4） |
| `loc_map_path` | 探针→源码行映射 JSON 路径 |
| `output_path` | 有效探针列表输出 JSON 路径 |
| `dead` | 是否执行死探针消除 |
| `merge` | 是否执行相邻探针合并 |

**死探针消除原理：** 找出在所有线程中 `start_idx` / `end_idx` 至少出现过一次的探针索引，过滤掉从未被任何线程执行的探针。

**相邻探针合并原理：** 分析相邻探针对之间的 `end_idx → next start_idx` 转移关系，若某对探针满足"1:1 唯一连接"且来自同一 Python 源码行，则将其合并（消除中间的出口/入口探针对），降低插桩粒度。

---

### PTX 探针模板（`template/`）

来自 cuTile 的现成探针，插桩时填入 INDEX 占位符后直接拼接进 PTX。

- `head.ptx`：在 kernel 入口声明探针所需的额外寄存器，并通过 `PARAM1`/`START`/`END`/`TOTAL` 占位符完成参数绑定（`PARAM1` 为 `time_buffer` 参数索引）
- `entry.ptx`：START 探针，读取 `%clock` 和 `%clock_hi`，将 `INDEX`（= `probe_idx << 16`）编码到 `start_hi` 高 16 位，暂存于寄存器
- `exit.ptx`：END 探针，同上读取 end 时钟，将 `[start_lo, start_hi, end_lo, end_hi]` 以 `st.global.v4.u32` 原子写入 `time_buffer`，并推进写指针
- `config.ptx`：线程过滤配置探针，通过 `PARAM` 占位符绑定 time buffer 参数索引

---

## 使用方法

### 1. 在 kernel 里添加一个 buffer 参数

```python
@triton.jit
def my_kernel(
    x_ptr, n,
    # 加在所有非 constexpr 参数的最后面
    time_buffer_ptr,
    BLOCK: tl.constexpr,
):
    ...
```

### 2. 用 `TritonInstrument` 上下文运行

`TritonInstrument` 是通用的，适用于任何 Triton kernel

```python
from tri_ins import TritonInstrument

with TritonInstrument(mode="block", t_start=0, t_end=127, output_dir="output") as inst:
    time_buf = inst.allocate_buffer(n_probes_estimate=128, n_threads=128)
    my_kernel[grid](..., time_buf, BLOCK=128)
    torch.cuda.synchronize()
    results = inst.get_results(export_path="output/raw")  # 同时保存 raw.npy 和 raw.csv
    inst.print_summary(results)
    inst.export_chrome_trace("output/trace.json", results)
```

完整可运行示例见 `examples/test_instrument.py`，直接执行即可：

```bash
python examples/test_instrument.py
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
| `analysis/loc_map.json` | 探针 ID → Python 源码行号的映射（插桩时生成） |
| `analysis/active_probes.json` | 经过死探针消除和合并后的有效探针列表（由 sanitizer 生成） |

---

