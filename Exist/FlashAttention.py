import math
import torch
import sys
import configparser
from math import ceil
import cuda.tile as ct
from cuda.tile import RoundingMode as RMd
import matplotlib.pyplot as plt
import numpy as np

torch.set_default_device(7)
INV_LOG_2 = 1.0 / math.log(2)

# Parameter Configuration
BATCH_SIZE = 4
NUM_HEADS = 128
NUM_HEAD_KV = 8
SEQ_LEN = 8192
HIDDEN_SIZE = 128

# Tile Configuration
TILE_M = 64
TILE_N = 64

NUM_THREAD = 128
BUFFER_LENGTH = 4096

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

@ct.kernel(num_ctas=1, occupancy=2)
def fmha_kernel(
    time_buffer,
    idx_buffer,
    Q,
    K,
    V,
    Out,
    qk_scale: float,
    input_pos: int,
    TILE_D: ConstInt,
    H: ConstInt,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    QUERY_GROUP_SIZE: ConstInt,
    CAUSAL: ConstBool,
    EVEN_K: ConstBool
):
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    batch_idx = bid_y // H
    head_idx = bid_y % H
    off_kv_h = head_idx // QUERY_GROUP_SIZE

    qk_scale = qk_scale * INV_LOG_2

    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    offs_m += input_pos
    offs_m = offs_m[:, None]

    offs_n_tile = ct.arange(TILE_N, dtype=ct.int32)
    offs_n_tile = offs_n_tile[None, :]

    m_i = ct.full((TILE_M, 1), -math.inf, dtype=ct.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

    q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D)).reshape((TILE_M, TILE_D))

    m_end = input_pos + (bid_x + 1) * TILE_M
    k_seqlen = K.shape[2]
    if CAUSAL:
        mask_start = (input_pos + bid_x * TILE_M) // TILE_N
        mask_start = min(mask_start, k_seqlen // TILE_N)
        Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)
    else:
        Tc = ct.cdiv(k_seqlen, TILE_N)
        mask_start = k_seqlen // TILE_N

    for j in range(0, Tc):
        k = ct.load(
            K,
            index=(batch_idx, off_kv_h, 0, j),
            shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),
            latency=2,
        )
        k = k.reshape((TILE_D, TILE_N))
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k, qk)

        if (CAUSAL or not EVEN_K) and j >= mask_start:
            offs_n = j * TILE_N + offs_n_tile
            mask = ct.full((TILE_M, TILE_N), True, dtype=ct.bool_)
            if not EVEN_K:
                mask = mask & (offs_n < k_seqlen)
            if CAUSAL:
                mask = mask & (offs_m >= offs_n)
            mask = ct.where(mask, 0.0, -math.inf)
            qk += mask

        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)
        qk = qk * qk_scale - m_ij

        p = ct.exp2(qk, flush_to_zero=True)
        l_ij = ct.sum(p, axis=-1, keepdims=True)
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        v = ct.load(
            V,
            index=(batch_idx, off_kv_h, j, 0),
            shape=(1, 1, TILE_N, TILE_D),
            latency=4,
        ).reshape((TILE_N, TILE_D))
        p = p.astype(Q.dtype)
        acc = ct.mma(p, v, acc)
        m_i = m_ij

    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)

def simple_cutile_fmha(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    input_pos: int = 0,
    causal: bool = True,
    export_path_raw: str = None,
    export_path_duration: str = None,
    time_buffer: torch.Tensor | None = None,
    idx_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    if not (Q.is_cuda and K.is_cuda and V.is_cuda):
        raise ValueError("Inputs must be CUDA tensors")
    if not (Q.device == K.device == V.device):
        raise ValueError("Inputs must be on the same device")
    
    batch_size, num_heads, seq_len, hidden_size = Q.shape
    _, num_head_kv, _, k_seq_len = K.shape
    
    # Calculate grid dimensions
    grid_x = ceil(seq_len / TILE_M)
    grid_y = batch_size * num_heads
    grid = (grid_x, grid_y, 1)
    
    Out = torch.empty_like(Q)
    qk_scale = 1.0 / math.sqrt(hidden_size)
    even_k = (k_seq_len % TILE_N == 0)
    query_group_size = num_heads // num_head_kv
    
    # Launch kernel
    ct.launch(
        torch.cuda.current_stream(), 
        grid, 
        fmha_kernel, 
        (time_buffer, idx_buffer, Q, K, V, Out, qk_scale, input_pos, HIDDEN_SIZE, num_heads, TILE_M, TILE_N, 
         query_group_size, causal, even_k)
    )
    
    torch.cuda.synchronize()
    
    idx_buffer_cpu = idx_buffer.cpu().numpy()
    time_buffer_cpu = time_buffer.cpu().numpy()

    # Merge idx and time data into buffer
    rows = len(idx_buffer_cpu) // (NUM_THREAD * 2)
    idx_buffer_2d = idx_buffer_cpu.reshape(rows, NUM_THREAD * 2)
    time_buffer_2d = time_buffer_cpu.reshape(rows, NUM_THREAD * 2)
    
    # Combine to raw format: each thread has 4 values [start_idx, end_idx, start_time, end_time]
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
    export_path_raw_csv = export_path_raw + ".csv"
    np.savetxt(export_path_raw_csv, buffer_2d_raw, delimiter=',', fmt='%d')
    print(f"已保存原始数据")

    # Calculate duration data
    if export_path_duration is not None:
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
        export_path_duration_csv = export_path_duration + ".csv"
        np.savetxt(export_path_duration_csv, duration_2d, delimiter=',', fmt='%d')
        print(f"已保存计时数据")
    
    return Out

if __name__ == "__main__":
    dtype = torch.float16
    
    config_path = "/home/junshan/cuTile/config.ini"
    config = configparser.ConfigParser()
    config.read(config_path)
    t_start = int(config['ominiscope']['t_start'])
    t_end = int(config['ominiscope']['t_end'])
    
    Q = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HIDDEN_SIZE, dtype=dtype, device="cuda")
    K = torch.randn(BATCH_SIZE, NUM_HEAD_KV, HIDDEN_SIZE, SEQ_LEN, dtype=dtype, device="cuda")
    V = torch.randn(BATCH_SIZE, NUM_HEAD_KV, SEQ_LEN, HIDDEN_SIZE, dtype=dtype, device="cuda")
    
    time_buffer = torch.zeros(NUM_THREAD * BUFFER_LENGTH * 2, dtype=torch.int64, device="cuda")
    idx_buffer = torch.zeros(NUM_THREAD * BUFFER_LENGTH * 2, dtype=torch.int16, device="cuda")
    
    export_path_raw = config['output']['raw_base'] + f"_{t_start}-{t_end}"
    export_path_duration = config['output']['duration_base'] + f"_{t_start}-{t_end}"
    
    Out = simple_cutile_fmha(
        Q, K, V, 
        input_pos=0, 
        causal=True,
        time_buffer=time_buffer, 
        idx_buffer=idx_buffer, 
        export_path_raw=export_path_raw, 
        export_path_duration=export_path_duration
    )
    
    print(f"Out shape={Out.shape}, dtype={Out.dtype}")
    print(f"BATCH_SIZE={BATCH_SIZE}, NUM_HEADS={NUM_HEADS}, NUM_HEAD_KV={NUM_HEAD_KV}")
    print(f"SEQ_LEN={SEQ_LEN}, HIDDEN_SIZE={HIDDEN_SIZE}, dtype={dtype}")