# Triton PTX Instrumentation

对 Triton kernel 编译产生的 PTX 进行自动插桩，在每个基本块的入口和出口插入时钟采样探针，从而获得 GPU 上各 sub-block 的执行耗时。

---

## 系统组成

```
Triton_insrt/
├── tri_ins/                   # 核心库
│   ├── __init__.py            # 对外暴露 TritonInstrument
│   ├── ptx_parser.py          # PTX 解析器：把 PTX 切分成 sub_block
│   └── triton_instrument.py   # 插桩引擎 + 结果收集
│
├── Exist_Package/             # PTX 探针模板（来自 cuTile）
│   ├── head.ptx               # 寄存器声明 / 初始化代码段
│   ├── entry.ptx              # 探针入口模板（START：记录进入时间戳）
│   ├── exit.ptx               # 探针出口模板（END：记录离开时间戳）
│   ├── config.ptx             # 探针配置（线程过滤等）
│   ├── insert_ptx.py          # cuTile 原始插桩脚本（参考实现）
│   ├── sub_block.py           # cuTile 原始 parser（参考实现）
│   └── ptxas                  # 备用 ptxas 可执行文件
│
├── examples/
│   └── test_instrument.py     # 端到端示例：matmul kernel 完整插桩流程
│
├── new_examples/              # 开发参考
│   ├── sub_block.py           # ptx_parser.py 的对照实现
│   ├── insert_ptx.py          # triton_instrument.py 的对照实现
│   ├── config.ini             # 配置文件示例
│   └── matmul.ptx             # cuTile 生成的参考插桩 PTX（用于验证）
│
└── output/                    # 运行时自动生成
    ├── original.ptx           # Triton 编译出的原始 PTX（插桩前）
    ├── instrumented.ptx       # 插桩后的 PTX
    ├── raw.csv                # 原始时间戳数据（每线程每探针）
    └── duration.csv           # 每探针耗时统计（ns）
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
| `mma_lines` | 含 `mma` 指令的行号列表 |
| `is_start` | 是否为 kernel 入口 `{` |
| `is_end` | 是否为 `ret;` |
| `is_sync` | 是否含有 `mbarrier.try_wait`（cuTile 异步等待块） |

**切块规则：**
- 遇到 `$L__BB` 标签 → 开启新 block
- 遇到无条件/条件跳转 → 关闭当前 block
- `bar.sync` 不触发切块，视为普通指令（Triton 特有行为）

---

### `tri_ins/triton_instrument.py`

核心插桩引擎，主要对外接口是 `TritonInstrument`。

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

**关键参数：**

| 参数 | 含义 | 默认值 |
|---|---|---|
| `mode` | `"block"`（per-block 计时）或 `"mma"`（per-mma 计时） | — |
| `t_start` / `t_end` | 采样线程的全局 thread ID 范围 | — |
| `target_file` | 只对 `.loc` 来自该文件索引的 block 插桩 | `1` |
| `output_dir` | PTX 文件保存目录 | `"output"` |

---

### PTX 探针模板（`Exist_Package/`）

来自 cuTile 的现成探针，插桩时填入 INDEX 占位符后直接拼接进 PTX。

- `head.ptx`：在 kernel 入口声明探针所需的额外寄存器（`%time_buf`、`%idx_buf` 等）
- `entry.ptx`：START 探针，用 `%clock64` 读当前时钟，写入 `time_buf[INDEX * N_THREADS + tid]`
- `exit.ptx`：END 探针，同上，写入 exit 时间戳

---

## 使用方法

### 1. 在 kernel 里添加两个 buffer 参数

```python
@triton.jit
def my_kernel(
    x_ptr, n,
    # 加在所有非 constexpr 参数的最后两个
    time_buffer_ptr,
    idx_buffer_ptr,
    BLOCK: tl.constexpr,
):
    ...
```

### 2. 用 `TritonInstrument` 上下文运行

`TritonInstrument` 是通用的，适用于任何 Triton kernel

```python
from tri_ins import TritonInstrument

with TritonInstrument(mode="block", t_start=0, t_end=127, output_dir="output") as inst:
    time_buf, idx_buf = inst.allocate_buffer(n_probes_estimate=128, n_threads=128)
    my_kernel[grid](..., time_buf, idx_buf, BLOCK=128)
    torch.cuda.synchronize()
    results = inst.get_results()
    inst.print_summary(results)
    inst.export_csv_raw("output/raw.csv", results)
    inst.export_csv_duration("output/duration.csv", results)
```

完整可运行示例见 `examples/test_instrument.py`，直接执行即可：

```bash
python examples/test_instrument.py
```

### 3. 查看输出文件

| 文件 | 内容 |
|---|---|
| `output/original.ptx` | 插桩前的原始 Triton PTX |
| `output/instrumented.ptx` | 插桩后的 PTX，可用于人工核查探针位置 |
| `output/raw.csv` | 每个线程在每个探针处的原始 `clock64` 时间戳 |
| `output/duration.csv` | 每个探针的 START→END 耗时（纳秒），按 thread 列出 |

---

