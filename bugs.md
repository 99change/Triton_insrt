# Bug 记录

## Bug #1: 插桩后 kernel 计算结果错误

### 现象

运行 `test_instrument.py`，插桩后的 matmul kernel 输出与 `torch.matmul` 的结果存在巨大偏差（diff 从几十到上万不等，且每次运行数值不同）。

```
Instrumented vs torch.matmul max diff: 6392.500000   # 每次不一样
```

不插桩时 diff = 0.0，说明 kernel 本身没问题。

### 根因

**Probe buffer 溢出。**

插桩在 PTX 中插入了 75 个 probe 点，每个 probe 的 Timer Stop 会执行 `st.global`（写入 buffer）并自增写指针：

```ptx
// Timer Stop: 每次触发都会推进写指针
st.global.u64 [%time_store_base], %start_time;
st.global.u64 [%time_store_base + 8], %end_time;
add.u64 %time_store_base, %time_store_base, %time_store_increment;
```

问题在于：75 个 probe 中有 **14 个位于 K-loop 内部**（`$L__BB0_3` 到 `@%p21 bra $L__BB0_3`）。对于 512×512×512 的 matmul（BLOCK_K=64），循环执行 **8 次**。

实际 probe 触发次数：

| 区域 | probe 数 | 执行次数 | 总触发 |
|------|----------|----------|--------|
| 循环外 | 61 | 1 | 61 |
| 循环内 | 14 | 8 | 112 |
| **合计** | 75 | — | **173** |

而 `allocate_buffer(n_probes_estimate=128)` 只分配了 128 个 probe 的空间。173 > 128，多出的 45 次 `st.global` 写到了 buffer 之外的 GPU 显存，覆盖了相邻的内存区域（包括输出矩阵 `c`），导致计算结果被污染。

### 修复

将 `n_probes_estimate` 从 128 改为 256：

```python
# 修复前
time_buf, idx_buf = inst.allocate_buffer(n_probes_estimate=128, n_threads=128)

# 修复后
time_buf, idx_buf = inst.allocate_buffer(n_probes_estimate=256, n_threads=128)
```

修复后 diff = 0.000000。

### 后续改进方向

当前的修复只是手动加大了 buffer。更好的做法：

1. **自动计算实际触发次数**：分析 PTX 中的循环结构，根据循环次数和循环内的 probe 数量自动计算所需 buffer 大小
2. **运行时边界检查**：在 probe 代码中加入地址范围判断，超出 buffer 范围时跳过写入
3. **不在循环内插 probe**：识别循环体，只在循环入口/出口插 probe，避免 buffer 用量随迭代次数线性增长
