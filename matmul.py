import torch
import sys
import configparser
from math import ceil
import cuda.tile as ct
import matplotlib.pyplot as plt
import numpy as np
torch.set_default_device(7)
ConstInt = ct.Constant[int]

def swizzle_2d(M, N, TILE_SIZE_M, TILE_SIZE_N, GROUP_SIZE_M):
    # Get the global IDs of the current CUDA block (CTA) in a 1D grid.
    bid = ct.bid(0)
    num_bid_m = ct.cdiv(M, TILE_SIZE_M)
    num_bid_n = ct.cdiv(N, TILE_SIZE_N)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + (bid % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m
    return bid_m, bid_n

@ct.kernel(num_ctas=1, occupancy=2)
def matmul_kernel(
    A,
    B,
    C,
    TILE_SIZE_M: ConstInt,
    TILE_SIZE_N: ConstInt,
    TILE_SIZE_K: ConstInt,
    time_buffer,
    idx_buffer
):
    GROUP_SIZE_M = 8
    M = A.shape[0]
    N = B.shape[1]
    bidx, bidy = swizzle_2d(M, N, TILE_SIZE_M, TILE_SIZE_N, GROUP_SIZE_M)
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(TILE_SIZE_M, TILE_SIZE_K))

    accumulator = ct.full((TILE_SIZE_M, TILE_SIZE_N), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    for k in range(num_tiles_k):
        a = ct.load(A, index=(bidx, k), shape=(TILE_SIZE_M, TILE_SIZE_K), padding_mode=zero_pad).astype(dtype)
        b = ct.load(B, index=(k, bidy), shape=(TILE_SIZE_K, TILE_SIZE_N), padding_mode=zero_pad).astype(dtype)
        accumulator = ct.mma(a, b, accumulator)

    accumulator = ct.astype(accumulator, C.dtype)
    ct.store(C, index=(bidx, bidy), tile=accumulator)

NUM_THREAD = 128
BUFFER_LENGTH = 330000
M = 8192
N = 16384
K = 53428
TM = 128
TN = 64
TK = 32

def simple_cutile_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    export_path_raw: str,
    export_path_duration: str,
    time_buffer: torch.Tensor | None = None,
    idx_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    if A.shape[1] != B.shape[0]:
        raise ValueError("K dimension mismatch")
    if not (A.is_cuda and B.is_cuda):
        raise ValueError("Inputs must be CUDA tensors")
    if A.device != B.device:
        raise ValueError("Inputs must be on the same device")

    m, _ = A.shape
    _, n = B.shape
    grid = (ceil(m / TM) * ceil(n / TN), 1, 1)

    C = torch.empty((m, n), device=A.device, dtype=A.dtype)

    ct.launch(torch.cuda.current_stream(), grid, matmul_kernel, (A, B, C, TM, TN, TK, time_buffer, idx_buffer))

    torch.cuda.synchronize()
    
    idx_buffer_cpu = idx_buffer.cpu().numpy()
    time_buffer_cpu = time_buffer.cpu().numpy()

    # 合并 idx 和 time 数据到一个 buffer
    rows = len(idx_buffer_cpu) // (NUM_THREAD * 2)
    idx_buffer_2d = idx_buffer_cpu.reshape(rows, NUM_THREAD * 2)
    time_buffer_2d = time_buffer_cpu.reshape(rows, NUM_THREAD * 2)
    
    # 组合成原始格式：每个线程 4 个值 [start_idx, end_idx, start_time, end_time]
    buffer_list_raw = []
    for row_idx in range(rows):
        row_data = []
        for thread_idx in range(NUM_THREAD):
            start_idx = idx_buffer_2d[row_idx, thread_idx * 2]
            end_idx = idx_buffer_2d[row_idx, thread_idx * 2 + 1]
            start_time = time_buffer_2d[row_idx, thread_idx * 2]
            end_time = time_buffer_2d[row_idx, thread_idx * 2 + 1]
            row_data.extend([start_idx, end_idx, start_time, end_time])
        buffer_list_raw.append(row_data)
    
    buffer_2d_raw = np.array(buffer_list_raw)
    export_path_raw_npy = export_path_raw + ".npy"
    np.save(export_path_raw_npy, buffer_2d_raw)
    export_path_raw = export_path_raw + ".csv"
    np.savetxt(export_path_raw, buffer_2d_raw, delimiter=',', fmt='%d')
    print(f"已保存原始数据")

    # 计算计时数据
    duration_list = []
    for row in buffer_2d_raw:
        row_duration = []
        for i in range(NUM_THREAD):
            start_idx = row[i * 4 + 0]
            end_idx = row[i * 4 + 1]
            start_time = row[i * 4 + 2]
            end_time = row[i * 4 + 3]
            block_duration = end_time - start_time
            row_duration.extend([start_idx, end_idx, block_duration])
        duration_list.append(row_duration)
    
    duration_2d = np.array(duration_list)
    print(f"第一个线程的总周期数: {np.sum(duration_2d[:, 2])}")
    export_path_duration_npy = export_path_duration + ".npy"
    np.save(export_path_duration_npy, duration_2d)
    export_path_duration = export_path_duration + ".csv"
    np.savetxt(export_path_duration, duration_2d, delimiter=',', fmt='%d')
    print(f"已保存计时数据")

    return C

if __name__ == "__main__":
    dtype = torch.float16

    config_path = "/home/junshan/cuTile/config.ini"
    config = configparser.ConfigParser()
    config.read(config_path)
    t_start = int(config['ominiscope']['t_start'])
    t_end = int(config['ominiscope']['t_end'])

    A = torch.randn(M, K, dtype=dtype, device="cuda")
    B = torch.randn(K, N, dtype=dtype, device="cuda")
    time_buffer = torch.zeros(NUM_THREAD * BUFFER_LENGTH * 2, dtype=torch.int64, device="cuda")
    idx_buffer = torch.zeros(NUM_THREAD * BUFFER_LENGTH * 2, dtype=torch.int16, device="cuda")
    export_path_raw = config['output']['raw_base'] + f"_{t_start}-{t_end}"
    export_path_duration = config['output']['duration_base'] + f"_{t_start}-{t_end}"

    C = simple_cutile_matmul(A, B, time_buffer=time_buffer, idx_buffer=idx_buffer, export_path_raw=export_path_raw, export_path_duration=export_path_duration)
    print(f"C shape={C.shape}, dtype={C.dtype}, M={M}, N={N}, K={K}, dtype={dtype}")