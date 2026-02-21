"""
Triton Kernel Instrumentation
==============================
Intercepts PTX during Triton compilation, inserts timing probes using the
existing cuTile probe templates, and collects per-block timing data.

Architecture:
  1. User adds `buffer_ptr` as the last non-constexpr param in their @triton.jit kernel
  2. TritonInstrument monkey-patches CUDABackend.make_cubin to intercept PTX
  3. PTX is parsed into sub-blocks, probes inserted at block entry/exit
  4. Modified PTX is compiled to cubin via ptxas (normal Triton path)
  5. At launch, user passes a pre-allocated torch.Tensor as buffer_ptr
  6. After sync, results are parsed from the tensor

Usage:
    import torch
    import triton
    from tri_ins import TritonInstrument

    # Add time_buffer_ptr and idx_buffer_ptr as the last two non-constexpr params
    @triton.jit
    def my_kernel(x_ptr, n, time_buffer_ptr, idx_buffer_ptr, BLOCK: tl.constexpr):
        ...

    with TritonInstrument(mode="block") as inst:
        time_buf, idx_buf = inst.allocate_buffer(n_probes_estimate=64, n_threads=128)
        my_kernel[grid](x, n, time_buf, idx_buf, BLOCK=128)
        torch.cuda.synchronize()
        results = inst.get_results()
"""

import os
import re
import json
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from .ptx_parser import sub_block, builder

# ---- Paths to PTX probe templates (from Exist_Package) ----
PACKAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'Exist_Package')


def _load_template(name: str) -> List[str]:
    """Load a PTX template file from Exist_Package."""
    path = os.path.join(PACKAGE_DIR, name)
    with open(path, 'r', encoding='utf-8') as f:
        return f.readlines()


def _replace_mark(lines: List[str], pattern: str, replacement: str) -> List[str]:
    """Replace all occurrences of pattern in lines (reuses Exist_Package logic)."""
    return [re.sub(pattern, replacement, line) for line in lines]


def _insert_lines(lines: List[str], line_num_1based: int, new_lines: List[str]) -> int:
    """Insert new_lines before the given 1-based line number. Returns count inserted."""
    idx = line_num_1based - 1
    for i, new_line in enumerate(new_lines):
        if not new_line.endswith('\n'):
            new_line += '\n'
        lines.insert(idx + i, new_line)
    return len(new_lines)


def _find_kernel_name(lines: List[str]) -> Optional[str]:
    """Extract kernel entry point name from PTX."""
    for line in lines:
        m = re.match(r'\s*\.visible\s+\.entry\s+(\w+)\s*\(', line)
        if m:
            return m.group(1)
    return None


def _find_buffer_param_indices(lines: List[str], kernel_name: str) -> Tuple[int, int]:
    """
    Find the parameter indices of time_buffer_ptr and idx_buffer_ptr in PTX.

    Triton appends 2 hidden params (global_scratch, profile_scratch) after all
    user params. The two user buffer params (time then idx) are at:
      time_param = max_param_index - 3
      idx_param  = max_param_index - 2
    """
    param_pattern = re.compile(rf'{re.escape(kernel_name)}_param_(\d+)')
    max_idx = -1
    for line in lines:
        m = param_pattern.search(line)
        if m:
            idx = int(m.group(1))
            max_idx = max(max_idx, idx)
    if max_idx < 3:
        raise ValueError(f"Found only {max_idx+1} params — need at least 4 "
                         f"(user params + time_buffer + idx_buffer + 2 hidden)")
    return max_idx - 3, max_idx - 2


def instrument_ptx(ptx_str: str,
                   mode: str = "block",
                   t_start: int = 0,
                   t_end: int = 127) -> Tuple[str, Dict[int, int], int]:
    """
    Insert instrumentation probes into a PTX string.

    Args:
        ptx_str:  Original PTX from Triton compilation
        mode:     "block" | "single" | "entire" | "config"
        t_start:  First global thread ID to probe
        t_end:    Last global thread ID to probe

    Returns:
        (modified_ptx_str, loc_map, n_probes)
        loc_map: {probe_id: python_source_line}
    """
    lines = [line + '\n' for line in ptx_str.split('\n')]

    kernel_name = _find_kernel_name(lines)
    if kernel_name is None:
        raise ValueError("No kernel entry point found in PTX")

    time_param, idx_param = _find_buffer_param_indices(lines, kernel_name)

    # Load probe templates from Exist_Package
    head_lines = _load_template('head.ptx')
    entry_lines = _load_template('entry.ptx')
    exit_lines = _load_template('exit.ptx')
    config_lines = _load_template('config.ptx')

    total = t_end - t_start + 1

    # Parameterize head template (PARAM1=time_buffer idx, PARAM2=idx_buffer idx)
    head_lines = _replace_mark(head_lines, r'KERNEL_NAME', kernel_name)
    head_lines = _replace_mark(head_lines, r'PARAM1', str(time_param))
    head_lines = _replace_mark(head_lines, r'PARAM2', str(idx_param))
    head_lines = _replace_mark(head_lines, r'START', str(t_start))
    head_lines = _replace_mark(head_lines, r'END', str(t_end))
    head_lines = _replace_mark(head_lines, r'TOTAL', str(total))

    # Parameterize config template (uses single PARAM = time_buffer)
    config_lines = _replace_mark(config_lines, r'KERNEL_NAME', kernel_name)
    config_lines = _replace_mark(config_lines, r'PARAM', str(time_param))

    # Parse PTX into sub-blocks
    sub_block_list = builder(lines)

    print(f"[Instrument] Kernel: {kernel_name}")
    print(f"[Instrument] time_param={time_param}, idx_param={idx_param}")
    print(f"[Instrument] Sub-blocks found: {len(sub_block_list)}")
    for blk in sub_block_list:
        blk.print_block()

    # Begin insertion
    loc_probe_map: Dict[int, int] = {}
    start_line = sub_block_list[0].entry
    offset = 0

    if mode in ("block", "single", "entire"):
        offset += _insert_lines(lines, start_line, head_lines)
    elif mode == "config":
        offset += _insert_lines(lines, start_line, config_lines)

    i = 1  # probe index (1-based)

    if mode == "block":
        for j, block in enumerate(sub_block_list):
            # skip control-flow markers and structural blocks
            if block.is_start or block.is_end or block.is_sync:
                continue
            # skip bare label lines ($L__BBN_N:) — they are branch targets, not
            # actual instructions; the probe on the preceding code block already
            # captures the timing boundary just before the branch.
            if block.is_label:
                continue
            loc_probe_map[i] = block.loc

            next_is_sync = (j + 1 < len(sub_block_list) and
                            sub_block_list[j + 1].is_sync)
            prev_is_sync = (j > 0 and sub_block_list[j - 1].is_sync)

            if next_is_sync:
                entry_insert = _replace_mark(entry_lines[:], r'INDEX', str(i))
                offset += _insert_lines(lines, block.entry + offset, entry_insert)
            elif prev_is_sync:
                exit_insert = _replace_mark(exit_lines[:], r'INDEX', str(i))
                offset += _insert_lines(lines, block.exit + offset + 1, exit_insert)
                i += 1
            else:
                entry_insert = _replace_mark(entry_lines[:], r'INDEX', str(i))
                exit_insert = _replace_mark(exit_lines[:], r'INDEX', str(i))
                offset += _insert_lines(lines, block.entry + offset, entry_insert)
                offset += _insert_lines(lines, block.exit + offset + 1, exit_insert)
                i += 1

    elif mode == "single":
        for block in sub_block_list:
            if block.is_start or block.is_end or block.is_sync:
                continue
            j = block.entry
            while j <= block.exit:
                if j in block.mma_lines:
                    if j - 5 not in block.mma_lines:
                        entry_insert = _replace_mark(entry_lines[:], r'INDEX', str(i))
                        offset += _insert_lines(lines, j + offset, entry_insert)
                        loc_probe_map[i] = block.loc
                        j += 5
                        continue
                    if j + 5 in block.mma_lines:
                        j += 5
                        continue
                    exit_insert = _replace_mark(exit_lines[:], r'INDEX', str(i))
                    offset += _insert_lines(lines, j + offset + 5, exit_insert)
                    i += 1
                    j += 5
                elif j in block.empty_lines:
                    j += 1
                else:
                    entry_insert = _replace_mark(entry_lines[:], r'INDEX', str(i))
                    exit_insert = _replace_mark(exit_lines[:], r'INDEX', str(i))
                    offset += _insert_lines(lines, j + offset, entry_insert)
                    offset += _insert_lines(lines, j + offset + 1, exit_insert)
                    loc_probe_map[i] = block.loc
                    i += 1
                    j += 1

    elif mode == "entire":
        entry_insert = _replace_mark(entry_lines[:], r'INDEX', str(i))
        exit_insert = _replace_mark(exit_lines[:], r'INDEX', str(i))
        offset += _insert_lines(lines, sub_block_list[1].entry + offset, entry_insert)
        offset += _insert_lines(lines, sub_block_list[-2].exit + offset + 1, exit_insert)
        loc_probe_map[i] = -1
        i += 1

    elif mode == "config":
        pass
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    n_probes = i - 1
    modified_ptx = ''.join(lines)

    return modified_ptx, loc_probe_map, n_probes


class TritonInstrument:
    """
    Context manager that instruments Triton kernels with timing probes.

    Monkey-patches CUDABackend.make_cubin to intercept PTX before compilation.
    The kernel must have `time_buffer_ptr` and `idx_buffer_ptr` as the last two
    non-constexpr parameters (before Triton's 2 hidden params).

    Example:
        with TritonInstrument(mode="block", t_start=0, t_end=127) as inst:
            time_buf, idx_buf = inst.allocate_buffer(n_probes_estimate=128, n_threads=128)
            kernel[grid](x, y, time_buf, idx_buf, BLOCK=128)
            torch.cuda.synchronize()
            results = inst.get_results()
    """

    def __init__(self, mode: str = "block", t_start: int = 0, t_end: int = 127):
        self.mode = mode
        self.t_start = t_start
        self.t_end = t_end
        self._original_make_cubin = None
        self._loc_map: Dict[int, int] = {}
        self._n_probes: int = 0
        self._n_threads: int = 0
        self._time_buffer: Optional[torch.Tensor] = None
        self._idx_buffer: Optional[torch.Tensor] = None
        self._active: bool = False
        self._dump_ptx: Optional[str] = None  # Path to dump instrumented PTX

    @property
    def time_buffer(self) -> Optional[torch.Tensor]:
        """The GPU timing buffer (int64). See allocate_buffer()."""
        return self._time_buffer

    @property
    def idx_buffer(self) -> Optional[torch.Tensor]:
        """The GPU index buffer (int16). See allocate_buffer()."""
        return self._idx_buffer

    @property
    def loc_map(self) -> Dict[int, int]:
        """Mapping from probe_id to Python source line number."""
        return self._loc_map

    @property
    def n_probes(self) -> int:
        """Number of probes inserted (available after first compilation)."""
        return self._n_probes

    def allocate_buffer(self, n_probes_estimate: int = 128,
                        n_threads: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pre-allocate the two output buffers for probe data.

        Buffer layout:
          time_buffer: int64, size = n_probes * n_threads * 2
            Per probe, per thread: [start_time, end_time]  (2 x 8 = 16 bytes)
            store_increment = n_threads * 16
          idx_buffer: int16, size = n_probes * n_threads * 2
            Per probe, per thread: [start_idx, end_idx]  (2 x 2 = 4 bytes)
            store_increment = n_threads * 4

        Args:
            n_probes_estimate: Estimated number of probes (overallocate is fine)
            n_threads: Number of probed threads (t_end - t_start + 1 by default)

        Returns:
            (time_buffer, idx_buffer) — pass both to the kernel as the last two params
        """
        if n_threads is None:
            n_threads = self.t_end - self.t_start + 1
        self._n_threads = n_threads
        n_elem = n_probes_estimate * n_threads * 2
        self._time_buffer = torch.zeros(n_elem, dtype=torch.int64, device='cuda')
        self._idx_buffer  = torch.zeros(n_elem, dtype=torch.int16, device='cuda')
        print(f"[Instrument] time_buffer: {n_elem * 8} bytes (int64)  "
              f"idx_buffer: {n_elem * 2} bytes (int16)  "
              f"({n_probes_estimate} probes x {n_threads} threads)")
        return self._time_buffer, self._idx_buffer

    def _hooked_make_cubin(self, original_fn, self_backend, src, metadata, opt, capability):
        """Intercept PTX, insert probes, then compile to cubin."""
        if self._active:
            try:
                modified_ptx, loc_map, n_probes = instrument_ptx(
                    src, mode=self.mode, t_start=self.t_start, t_end=self.t_end
                )
                self._loc_map = loc_map
                self._n_probes = n_probes

                print(f"\n[Instrument] Inserted {n_probes} probes, mode={self.mode}")
                print(f"[Instrument] Thread range: [{self.t_start}, {self.t_end}]")
                print(f"[Instrument] Loc map: {json.dumps(loc_map, indent=2)}")

                if self._dump_ptx:
                    with open(self._dump_ptx, 'w') as f:
                        f.write(modified_ptx)
                    print(f"[Instrument] Dumped instrumented PTX to {self._dump_ptx}")

                src = modified_ptx
            except Exception as e:
                print(f"[Instrument] Instrumentation failed: {e}")
                import traceback
                traceback.print_exc()
                print("[Instrument] Falling back to original PTX")

        return original_fn(self_backend, src, metadata, opt, capability)

    def __enter__(self):
        """Install the PTX interception hook."""
        from triton.backends.nvidia.compiler import CUDABackend
        self._original_make_cubin = CUDABackend.make_cubin

        instrument = self
        original = self._original_make_cubin

        def hooked(backend_self, src, metadata, opt, capability):
            return instrument._hooked_make_cubin(
                original, backend_self, src, metadata, opt, capability
            )

        CUDABackend.make_cubin = hooked
        self._active = True
        return self

    def __exit__(self, *args):
        """Restore the original make_cubin."""
        from triton.backends.nvidia.compiler import CUDABackend
        CUDABackend.make_cubin = self._original_make_cubin
        self._active = False

    def get_results(self) -> Dict[int, dict]:
        """
        Parse the two buffer tensors and return structured timing data.

        Returns:
            Dict mapping probe_id -> {
                'loc': int,           # Python source line number
                'start_idx': Tensor,  # probe start index per thread (int16)
                'end_idx': Tensor,    # probe end index per thread (int16)
                'start_time': Tensor, # clock64 at entry (int64)
                'end_time': Tensor,   # clock64 at exit (int64)
                'duration': Tensor,   # end_time - start_time (int64)
            }
        """
        if self._time_buffer is None or self._idx_buffer is None:
            raise RuntimeError("No buffers allocated. Call allocate_buffer() first.")

        n_threads = self._n_threads or (self.t_end - self.t_start + 1)
        n_probes = self._n_probes
        if n_probes == 0:
            print("[Instrument] Warning: n_probes=0, no probes were inserted")
            return {}

        # Reshape: [n_probes, n_threads, 2]
        time_data = self._time_buffer[:n_probes * n_threads * 2].reshape(n_probes, n_threads, 2).cpu()
        idx_data  = self._idx_buffer[:n_probes * n_threads * 2].reshape(n_probes, n_threads, 2).cpu()

        results = {}
        for probe_idx in range(n_probes):
            probe_id = probe_idx + 1
            loc        = self._loc_map.get(probe_id, -1)
            start_time = time_data[probe_idx, :, 0]
            end_time   = time_data[probe_idx, :, 1]
            start_idx  = idx_data[probe_idx, :, 0]
            end_idx    = idx_data[probe_idx, :, 1]
            duration   = end_time - start_time
            results[probe_id] = {
                'loc': loc,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'mean_cycles': duration.float().mean().item(),
            }

        return results

    def print_summary(self, results: Optional[Dict] = None):
        """Print a human-readable summary of instrumentation results."""
        if results is None:
            results = self.get_results()

        print("\n" + "=" * 70)
        print("Triton Kernel Instrumentation Results")
        print("=" * 70)
        print(f"{'Probe':>6} {'Loc':>6} {'Mean Cycles':>14} {'Min':>12} {'Max':>12}")
        print("-" * 70)
        for probe_id, data in sorted(results.items()):
            dur = data['duration']
            nonzero = dur[dur > 0]
            if len(nonzero) > 0:
                mean = nonzero.float().mean().item()
                mn = nonzero.min().item()
                mx = nonzero.max().item()
            else:
                mean = mn = mx = 0
            print(f"{probe_id:>6} {data['loc']:>6} {mean:>14.1f} {mn:>12} {mx:>12}")
        print("=" * 70)

    def export_csv_raw(self, path: str, results: Optional[Dict] = None):
        """
        Export raw per-probe per-thread data as CSV.

        Matches cuTile matmul.py's buffer_list_raw format:
          rows    = probes (one row per probe)
          columns = for each thread: [start_idx, end_idx, start_time, end_time]

        Header: probe_id, loc, t0_start_idx, t0_end_idx, t0_start_time, t0_end_time, ...
        """
        if results is None:
            results = self.get_results()
        n_threads = self._n_threads or (self.t_end - self.t_start + 1)

        header = ['probe_id', 'loc']
        for t in range(n_threads):
            header += [f't{t}_start_idx', f't{t}_end_idx',
                       f't{t}_start_time', f't{t}_end_time']

        rows = []
        for probe_id, data in sorted(results.items()):
            row = [probe_id, data['loc']]
            for t in range(n_threads):
                row += [
                    data['start_idx'][t].item(),
                    data['end_idx'][t].item(),
                    data['start_time'][t].item(),
                    data['end_time'][t].item(),
                ]
            rows.append(row)

        arr = np.array(rows, dtype=np.int64)
        np.savetxt(path, arr, delimiter=',', fmt='%d',
                   header=','.join(header), comments='')
        print(f"[Instrument] Raw CSV saved: {path}  ({len(rows)} probes x {n_threads} threads)")

    def export_csv_duration(self, path: str, results: Optional[Dict] = None):
        """
        Export per-thread timing summary as CSV.

        Output format (matches cuTile's duration CSV, transposed to thread-centric):
          rows    = threads  (index = thread_id within the probed range)
          columns = per-probe duration [cycles] + total_cycles

        Header: thread_id, probe_1_loc<N>, probe_2_loc<N>, ..., total_cycles
        """
        if results is None:
            results = self.get_results()
        n_threads = self._n_threads or (self.t_end - self.t_start + 1)
        sorted_probes = sorted(results.keys())

        header = ['thread_id']
        for pid in sorted_probes:
            header.append(f'probe_{pid}_loc{results[pid]["loc"]}')
        header.append('total_cycles')

        rows = []
        for t in range(n_threads):
            row = [self.t_start + t]
            total = 0
            for pid in sorted_probes:
                d = results[pid]['duration'][t].item()
                # negative values mean exit probe wasn't reached (branch taken)
                d = max(d, 0)
                row.append(d)
                total += d
            row.append(total)
            rows.append(row)

        arr = np.array(rows, dtype=np.int64)
        np.savetxt(path, arr, delimiter=',', fmt='%d',
                   header=','.join(header), comments='')
        print(f"[Instrument] Duration CSV saved: {path}  ({n_threads} threads x {len(sorted_probes)} probes)")
        if len(rows) > 0:
            totals = arr[:, -1]
            print(f"[Instrument] Total cycles — mean: {totals.mean():.0f}  "
                  f"min: {totals.min()}  max: {totals.max()}")
