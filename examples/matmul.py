"""
Triton Matrix Multiplication
=============================
Based on the official Triton tutorial:
https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html

Computes C = A @ B where A is (M, K) and B is (K, N).
Uses block-level tiling with L2 cache optimization via "grouped ordering".
"""

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides (number of elements to skip to move by 1 in that dimension)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.

    A has shape (M, K), B has shape (K, N) and C has shape (M, N).
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Optional fused activation
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)

    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


def matmul(a, b, activation=""):
    """Triton matrix multiplication wrapper."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"

    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=256,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=8,
        ACTIVATION=activation,
    )
    return c


# ---------------------------------------------------------------------------
# Test & Benchmark
# ---------------------------------------------------------------------------

def test_matmul():
    """Correctness test: compare Triton matmul vs torch.matmul."""
    torch.manual_seed(0)
    a = torch.randn((512, 512), device="cuda", dtype=torch.float16)
    b = torch.randn((512, 512), device="cuda", dtype=torch.float16)

    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)

    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
        print(f"✅ Triton matmul matches torch.matmul (max diff = "
              f"{(triton_output - torch_output).abs().max().item():.6f})")
    else:
        print(f"❌ Mismatch! max diff = "
              f"{(triton_output - torch_output).abs().max().item():.6f}")

    # Also test with activation
    triton_act = matmul(a, b, activation="leaky_relu")
    torch_act = torch.matmul(a, b)
    torch_act = torch.where(torch_act >= 0, torch_act, 0.01 * torch_act)
    if torch.allclose(triton_act, torch_act, atol=1e-2, rtol=0):
        print(f"✅ Triton matmul+leaky_relu matches (max diff = "
              f"{(triton_act - torch_act).abs().max().item():.6f})")
    else:
        print(f"❌ Mismatch with activation! max diff = "
              f"{(triton_act - torch_act).abs().max().item():.6f}")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[128 * i for i in range(2, 33)],
        line_arg="provider",
        line_vals=["cublas", "triton"],
        line_names=["cuBLAS", "Triton"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",
        plot_name="matmul-performance",
        args={},
    )
)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "cublas":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.matmul(a, b), quantiles=quantiles
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul(a, b), quantiles=quantiles
        )
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    print("=" * 60)
    print("Triton Matrix Multiplication - Correctness Test")
    print("=" * 60)
    test_matmul()

    print()
    print("=" * 60)
    print("Triton Matrix Multiplication - Benchmark")
    print("=" * 60)
    try:
        benchmark.run(show_plots=False, print_data=True)
    except ImportError:
        print("(matplotlib not installed, running manual benchmark instead)")
        for size in [512, 1024, 2048]:
            a = torch.randn((size, size), device="cuda", dtype=torch.float16)
            b = torch.randn((size, size), device="cuda", dtype=torch.float16)

            cublas_ms = triton.testing.do_bench(lambda: torch.matmul(a, b))
            triton_ms = triton.testing.do_bench(lambda: matmul(a, b))
            tflops = lambda ms: 2 * size**3 * 1e-12 / (ms * 1e-3)
            print(f"  [{size:4d}x{size:4d}]  cuBLAS: {tflops(cublas_ms):6.1f} TFLOPS ({cublas_ms:.3f} ms)"
                  f"  |  Triton: {tflops(triton_ms):6.1f} TFLOPS ({triton_ms:.3f} ms)")
