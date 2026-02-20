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

    print("=" * 70)
    print("Phase 1: Correctness check (without instrumentation)")
    print("=" * 70)

    # Run uninstrumented kernel to get reference output
    c_ref = torch.empty((M, N), device="cuda", dtype=torch.float16)
    _dummy = torch.zeros(1, dtype=torch.int64, device="cuda")
    matmul_kernel[grid](
        a, b, c_ref,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c_ref.stride(0), c_ref.stride(1),
        _dummy,  # dummy time_buffer
        _dummy,  # dummy idx_buffer
        BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K, GROUP_SIZE_M=8,
    )
    torch.cuda.synchronize()
    torch_ref = torch.matmul(a, b)
    print(f"Max diff vs torch.matmul: {(c_ref - torch_ref).abs().max().item():.6f}")

    print("\n" + "=" * 70)
    print("Phase 2: Instrumented run")
    print("=" * 70)

    # Force recompilation: clear in-memory cache AND disk cache bypass
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    device = torch.cuda.current_device()
    matmul_kernel.device_caches[device][0].clear()  # clear in-memory kernel cache

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Probe the first block's threads: global_tid 0..127
    with TritonInstrument(mode="block", t_start=0, t_end=127) as inst:
        inst._dump_ptx = os.path.join(output_dir, "instrumented.ptx")

        # Allocate both buffers (generous estimate: 128 probes, 128 threads)
        time_buf, idx_buf = inst.allocate_buffer(n_probes_estimate=128, n_threads=128)

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

        # Check correctness: instrumented kernel should produce same output
        max_diff = (c_inst - c_ref).abs().max().item()
        print(f"\nInstrumented vs reference max diff: {max_diff:.6f}")

        # Print timing results
        results = inst.get_results()
        inst.print_summary(results)

    # Unset to avoid affecting other compilations
    os.environ.pop("TRITON_ALWAYS_COMPILE", None)


if __name__ == "__main__":
    main()
