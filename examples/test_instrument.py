"""
End-to-end test: Triton matmul with PTX instrumentation
=========================================================
1. Defines a matmul kernel with an extra `buffer_ptr` parameter
2. Uses TritonInstrument to intercept PTX and insert timing probes
3. Runs the kernel, collects timing data, and prints results
"""

import os
import sys
import torch
import triton
import triton.language as tl

# Add project root so `tri_ins` is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tri_ins import TritonInstrument


@triton.jit
def matmul_kernel(
    # Original params
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # === Instrumentation buffers (last two non-constexpr params) ===
    time_buffer_ptr,
    idx_buffer_ptr,
    # Constexpr params (not in PTX)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Matmul kernel with buffer_ptr for instrumentation."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def main():
    torch.manual_seed(0)

    M, N, K = 512, 512, 512
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64

    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    # grid = (4*2,) = (8,) blocks, each with 128 threads
    # Total threads = 8 * 128 = 1024, global thread IDs [0, 1023]

    # Ground truth from PyTorch
    torch_ref = torch.matmul(a, b)

    print("=" * 70)
    print("Instrumented run")
    print("=" * 70)

    # Load settings from tri_ins/config.ini
    inst = TritonInstrument.from_config()

    output_dir = os.path.abspath(inst.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    # Update output_dir to absolute path so PTX saves go to the right place
    inst.output_dir = output_dir

    n_threads = inst.t_end - inst.t_start + 1

    with inst:
        time_buf, idx_buf = inst.allocate_buffer(
            n_probes_estimate=inst.n_probes_estimate, n_threads=n_threads)

        # Launch instrumented kernel
        c_inst = torch.empty((M, N), device="cuda", dtype=torch.float16)
        matmul_kernel[grid](
            a, b, c_inst,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c_inst.stride(0), c_inst.stride(1),
            time_buf,  # ← timing buffer (int64)
            idx_buf,   # ← index buffer (int16)
            BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=BLOCK_N,
            BLOCK_SIZE_K=BLOCK_K, GROUP_SIZE_M=8,
        )
        torch.cuda.synchronize()

        # Check correctness against torch.matmul
        max_diff = (c_inst.float() - torch_ref.float()).abs().max().item()
        print(f"\nInstrumented vs torch.matmul max diff: {max_diff:.6f}")

        # Print timing results
        results = inst.get_results()
        inst.print_summary(results)

        # Export CSV results
        inst.export_csv_raw(os.path.join(output_dir, inst.raw_csv), results)

        # Export Chrome trace JSON
        inst.export_chrome_trace(os.path.join(output_dir, inst.trace_json), results)


if __name__ == "__main__":
    main()
