"""
Dump PTX from Triton matmul kernel and inspect .loc directives.

Triton's compilation pipeline:
  Python AST → TTIR → TTGIR → LLVM IR → PTX → cubin

By default, Triton runs `add_di_scope` pass which injects debug info
(Python source file + line numbers) into LLVM IR. When LLVM translates
to PTX, these become `.loc` and `.file` directives.

Key env vars:
  - TRITON_DISABLE_LINE_INFO=1  → removes .loc (we do NOT want this)
  - TRITON_KERNEL_DUMP=1        → dumps all IR stages to disk
  - USE_IR_LOC=ptx              → replaces locations with IR-level locations
"""

import os
import torch
import triton
import triton.language as tl


# ---- Simple kernel for testing ----
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
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
    # --- Step 1: Compile the kernel and get the CompiledKernel ---
    M, N, K = 512, 512, 512
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    # Force compilation by running the kernel once
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        GROUP_SIZE_M=8,
    )
    torch.cuda.synchronize()

    # --- Step 2: Extract PTX from the compiled kernel cache ---
    # Access the compiled kernel from the cache
    device = torch.cuda.current_device()
    kernel_cache = matmul_kernel.device_caches[device][0]

    print("=" * 70)
    print("Compiled kernels in cache:", len(kernel_cache))
    print("=" * 70)

    for key, compiled_kernel in kernel_cache.items():
        ptx = compiled_kernel.asm.get("ptx", None)
        if ptx is None:
            print("No PTX found in compiled kernel!")
            continue

        # Save PTX to file
        ptx_path = "/home/zhaoling/WORKSPACE/Triton_insrt/matmul_kernel.ptx"
        with open(ptx_path, "w") as f:
            f.write(ptx)
        print(f"PTX saved to: {ptx_path}")
        print(f"PTX length: {len(ptx)} chars, {ptx.count(chr(10))} lines")

        # --- Step 3: Analyze .loc and .file directives ---
        lines = ptx.split("\n")
        file_directives = [l.strip() for l in lines if l.strip().startswith(".file")]
        loc_directives = [l.strip() for l in lines if l.strip().startswith(".loc")]

        print(f"\n--- .file directives ({len(file_directives)}) ---")
        for fd in file_directives:
            print(f"  {fd}")

        print(f"\n--- .loc directives ({len(loc_directives)} total) ---")
        # Show unique .loc patterns
        unique_locs = sorted(set(loc_directives))
        print(f"    Unique .loc entries: {len(unique_locs)}")
        for loc in unique_locs[:30]:  # Show first 30
            print(f"  {loc}")
        if len(unique_locs) > 30:
            print(f"  ... ({len(unique_locs) - 30} more)")

        # --- Step 4: Show PTX snippet around a .loc ---
        print(f"\n--- Sample PTX with .loc context ---")
        for i, line in enumerate(lines):
            if ".loc" in line and i < len(lines) - 3:
                for j in range(max(0, i-1), min(len(lines), i+4)):
                    print(f"  {j+1:4d}: {lines[j]}")
                print("  ...")
                break

        # Print first 80 lines of PTX for overview
        print(f"\n--- PTX header (first 80 lines) ---")
        for i, line in enumerate(lines[:80]):
            print(f"  {i+1:4d}: {line}")


if __name__ == "__main__":
    main()
