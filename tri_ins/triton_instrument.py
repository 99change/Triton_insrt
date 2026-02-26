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
import configparser
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from .ptx_parser import sub_block, builder

# Default config path (alongside this module)
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')

# ---- Paths to PTX probe templates ----
PACKAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'template')


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
                   t_end: int = 127,
                   target_file: int = 1) -> Tuple[str, Dict[int, int], int]:
    """
    Insert instrumentation probes into a PTX string.

    Args:
        ptx_str:     Original PTX from Triton compilation
        mode:        "block" | "single" | "entire" | "config"
        t_start:     First global thread ID to probe
        t_end:       Last global thread ID to probe
        target_file: Source file index to instrument (from .loc directive).
                     Triton PTX can reference multiple source files; use this
                     to select which one corresponds to your kernel file.
                     Run with any PTX and inspect the printed sub-blocks to
                     find the right index (default=1).

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
    print(f"[Instrument] Sub-blocks found: {len(sub_block_list)}  (target_file={target_file})")
    file_indices = sorted({b.file for b in sub_block_list})
    print(f"[Instrument] Source file indices seen in .loc: {file_indices}")
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

    i = 0  # probe index (0-based, matches insert_ptx.py)

    if mode == "block":
        for j, block in enumerate(sub_block_list):
            # skip control-flow markers and structural blocks
            if block.is_start or block.is_end or block.is_sync:
                continue
            # only instrument blocks that belong to the target source file
            if block.file != target_file:
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

    n_probes = i  # i was incremented once per probe pair, so i == total probes
    modified_ptx = ''.join(lines)

    return modified_ptx, loc_probe_map, n_probes


class TritonInstrument:
    """
    Context manager that instruments Triton kernels with timing probes.

    Monkey-patches CUDABackend.make_cubin to intercept PTX before compilation.
    The kernel must have `time_buffer_ptr` and `idx_buffer_ptr` as the last two
    non-constexpr parameters (before Triton's 2 hidden params).

    Args:
        mode:        Instrumentation mode: "block" | "single" | "entire" | "config"
        t_start:     First global thread ID to probe
        t_end:       Last global thread ID to probe
        target_file: Source file index to target (from PTX .loc directives).
                     Triton PTX can embed code from multiple source files
                     (e.g. your kernel + triton stdlib helpers). The first run
                     will print all file indices found — use that to pick the
                     right one (default=1, which is typically the user kernel).
        output_dir:  Directory where both PTX files are always saved.
                     - ``<output_dir>/original.ptx``     — PTX before instrumentation
                     - ``<output_dir>/instrumented.ptx`` — PTX after instrumentation
                     The directory is created automatically if it does not exist.
                     Pass ``None`` to disable saving.

    Example:
        with TritonInstrument(mode="block", t_start=0, t_end=127, target_file=1,
                              output_dir="output") as inst:
            time_buf, idx_buf = inst.allocate_buffer(n_probes_estimate=128, n_threads=128)
            kernel[grid](x, y, time_buf, idx_buf, BLOCK=128)
            torch.cuda.synchronize()
            results = inst.get_results()
    """

    def __init__(self, mode: str = "block", t_start: int = 0, t_end: int = 127,
                 target_file: int = 1,
                 output_dir: Optional[str] = "output",
                 n_probes_estimate: int = 256,
                 raw_csv: str = "raw.csv",
                 trace_json: str = "trace.json",
                 loc_map_json: str = "loc_map.json",
                 dump_ptx: bool = True):
        self.mode = mode
        self.t_start = t_start
        self.t_end = t_end
        self.target_file = target_file
        self.output_dir = output_dir
        self.n_probes_estimate = n_probes_estimate
        self.raw_csv = raw_csv
        self.trace_json = trace_json
        self.loc_map_json = loc_map_json
        self.dump_ptx = dump_ptx
        self._original_make_cubin = None
        self._loc_map: Dict[int, int] = {}
        self._n_probes: int = 0
        self._n_threads: int = 0
        self._time_buffer: Optional[torch.Tensor] = None
        self._idx_buffer: Optional[torch.Tensor] = None
        self._active: bool = False

    @classmethod
    def from_config(cls, config_path: str = None) -> 'TritonInstrument':
        """
        Create a TritonInstrument from an INI config file.

        If config_path is None, uses tri_ins/config.ini.
        """
        if config_path is None:
            config_path = DEFAULT_CONFIG
        cfg = configparser.ConfigParser()
        cfg.read(config_path)

        inst_sec = cfg['instrument']
        out_sec = cfg['output']

        return cls(
            mode=inst_sec.get('mode', 'block'),
            t_start=inst_sec.getint('t_start', 0),
            t_end=inst_sec.getint('t_end', 127),
            target_file=inst_sec.getint('target_file', 1),
            n_probes_estimate=inst_sec.getint('n_probes_estimate', 256),
            output_dir=out_sec.get('output_dir', 'output'),
            raw_csv=out_sec.get('raw_csv', 'raw.csv'),
            trace_json=out_sec.get('trace_json', 'trace.json'),
            loc_map_json=out_sec.get('loc_map', 'loc_map.json'),
            dump_ptx=out_sec.getboolean('dump_ptx', True),
        )

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
                # Save original PTX when dump_ptx is enabled
                if self.output_dir and self.dump_ptx:
                    os.makedirs(self.output_dir, exist_ok=True)
                    orig_path = os.path.join(self.output_dir, "original.ptx")
                    with open(orig_path, 'w') as f:
                        f.write(src)
                    print(f"[Instrument] Saved original PTX  -> {orig_path}")

                modified_ptx, loc_map, n_probes = instrument_ptx(
                    src, mode=self.mode, t_start=self.t_start, t_end=self.t_end,
                    target_file=self.target_file
                )
                self._loc_map = loc_map
                self._n_probes = n_probes

                print(f"\n[Instrument] Inserted {n_probes} probes, mode={self.mode}")
                print(f"[Instrument] Thread range: [{self.t_start}, {self.t_end}]")
                print(f"[Instrument] Loc map: {json.dumps(loc_map, indent=2)}")

                # Save instrumented PTX when dump_ptx is enabled
                if self.output_dir and self.dump_ptx:
                    inst_path = os.path.join(self.output_dir, "instrumented.ptx")
                    with open(inst_path, 'w') as f:
                        f.write(modified_ptx)
                    print(f"[Instrument] Saved instrumented PTX -> {inst_path}")

                # Save loc→probe mapping as JSON
                if self.output_dir:
                    loc_path = os.path.join(self.output_dir, self.loc_map_json)
                    with open(loc_path, 'w') as f:
                        json.dump(loc_map, f, indent=2)
                    print(f"[Instrument] Saved loc map -> {loc_path}")

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
            probe_id = probe_idx      # display key matches 0-based PTX INDEX
            loc        = self._loc_map.get(probe_idx, -1)
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

        Format:
          rows    = probes (one row per probe)
          columns = for each thread (1-based): start_idx, end_idx, start_time, end_time

        Header: t1_start_idx, t1_end_idx, t1_start_time, t1_end_time, t2_..., ...
        """
        if results is None:
            results = self.get_results()
        n_threads = self._n_threads or (self.t_end - self.t_start + 1)

        header = []
        for t in range(n_threads):
            header += [f't{t}_start_idx', f't{t}_end_idx',
                       f't{t}_start_time', f't{t}_end_time']

        rows = []
        for probe_id, data in sorted(results.items()):
            row = []
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

    def export_chrome_trace(self, path: str, results: Optional[Dict] = None,
                            chunk_size: int = 4096, workers: int = 16,
                            queue_size: int = 16):
        """
        Export probe data as Chrome Trace Event JSON.

        Output can be opened in chrome://tracing or https://ui.perfetto.dev.
        Each probe becomes an "X" (complete) event with:
          - ts   = start_time (clock cycles)
          - dur  = end_time - start_time
          - tid  = thread index within probed range
          - pid  = 0
          - name = "Line <loc>"  (Python source line from .loc mapping)
        """
        if results is None:
            results = self.get_results()
        n_threads = self._n_threads or (self.t_end - self.t_start + 1)
        loc_map = self._loc_map

        # Build raw data array: [n_probes, n_threads * 4]
        rows = []
        for probe_id, data in sorted(results.items()):
            row = []
            for t in range(n_threads):
                row += [
                    data['start_idx'][t].item(),
                    data['end_idx'][t].item(),
                    data['start_time'][t].item(),
                    data['end_time'][t].item(),
                ]
            rows.append(row)
        data_arr = np.array(rows, dtype=np.int64)

        if data_arr.size == 0:
            with open(path, 'w') as f:
                json.dump({"schemaVersion": 1, "traceEvents": [],
                           "displayTimeUnit": "ns"}, f)
            print(f"[Instrument] Chrome trace saved: {path}  (empty)")
            return

        q: "queue.Queue[object]" = queue.Queue(maxsize=queue_size)

        def worker(start_row: int, end_row: int):
            parts = []
            for row in data_arr[start_row:end_row]:
                for i in range(n_threads):
                    start_idx  = int(row[i * 4])
                    end_idx    = int(row[i * 4 + 1])
                    start_time = int(row[i * 4 + 2])
                    end_time   = int(row[i * 4 + 3])

                    if start_time == 0:
                        continue

                    duration = end_time - start_time
                    loc = loc_map.get(start_idx, None)
                    name = f"Line {loc}" if loc is not None else f"Probe {start_idx}"

                    event = {
                        "name": name,
                        "cat": name,
                        "ph": "X",
                        "ts": start_time,
                        "dur": duration,
                        "pid": 0,
                        "tid": i,
                        "args": {
                            "start_index": start_idx,
                            "end_index": end_idx
                        }
                    }
                    parts.append(json.dumps(event, ensure_ascii=False))

            if parts:
                q.put(',\n'.join(parts))

        def writer_fn():
            first = True
            with open(path, 'w', encoding='utf-8') as f:
                f.write('{"schemaVersion": 1, "traceEvents": [\n')
                while True:
                    item = q.get()
                    if item is None:
                        break
                    if not first:
                        f.write(',\n')
                    f.write(item)
                    first = False
                f.write('\n], "displayTimeUnit": "ns"}')

        total_rows = data_arr.shape[0]

        writer = threading.Thread(target=writer_fn, daemon=True)
        writer.start()

        ranges = [(s, min(s + chunk_size, total_rows))
                  for s in range(0, total_rows, chunk_size)]

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(worker, s, e) for s, e in ranges]
            for fut in futures:
                fut.result()

        q.put(None)
        writer.join()

        print(f"[Instrument] Chrome trace saved: {path}  "
              f"({len(rows)} probes x {n_threads} threads)")

    # ------------------------------------------------------------------ #
    #  Sanitizer: dead-probe elimination + merge
    # ------------------------------------------------------------------ #

    def _build_raw_array(self, results: Optional[Dict] = None) -> np.ndarray:
        """Build the raw numpy array [n_probes, n_threads * 4] from results."""
        if results is None:
            results = self.get_results()
        n_threads = self._n_threads or (self.t_end - self.t_start + 1)
        rows = []
        for probe_id, data in sorted(results.items()):
            row = []
            for t in range(n_threads):
                row += [
                    data['start_idx'][t].item(),
                    data['end_idx'][t].item(),
                    data['start_time'][t].item(),
                    data['end_time'][t].item(),
                ]
            rows.append(row)
        return np.array(rows, dtype=np.int64)

    @staticmethod
    def _clear_dead(data: np.ndarray) -> Tuple[List[int], List[int]]:
        """Find probes that were actually executed (non-dead)."""
        start_idx = data[:, ::4]
        end_idx = data[:, 1::4]
        active_start = sorted(set(np.unique(start_idx).tolist()))
        active_end = sorted(set(np.unique(end_idx).tolist()))
        return active_start, active_end

    @staticmethod
    def _get_connections(data: np.ndarray) -> set:
        """Find (end_idx → next start_idx) transitions across consecutive probes."""
        start_index = data[:, ::4]
        end_index = data[:, 1::4]
        # Only count transitions where a chain of non-zero start_idx continues
        valid = np.cumprod(start_index[1:] != 0, axis=0).astype(bool)
        from_nodes = end_index[:-1][valid]
        to_nodes = start_index[1:][valid]
        return set(zip(from_nodes.tolist(), to_nodes.tolist()))

    @staticmethod
    def _get_merge_list(connections: set, loc_map: Dict[int, int]) -> List[Tuple[int, int]]:
        """Find probe pairs that can be merged (unique 1:1 connection, same loc)."""
        merged = []
        for from_idx, to_idx in connections:
            is_unique_from = sum(1 for c in connections if c[0] == from_idx) == 1
            is_unique_to = sum(1 for c in connections if c[1] == to_idx) == 1
            if is_unique_from and is_unique_to:
                from_loc = loc_map.get(from_idx, None)
                to_loc = loc_map.get(to_idx, None)
                if from_loc is not None and from_loc == to_loc:
                    merged.append((from_idx, to_idx))
        return merged

    def sanitize(self, path: str, results: Optional[Dict] = None,
                 dead: bool = True, merge: bool = True):
        """
        Perform dead-probe elimination and/or merge, output active probe list.

        Args:
            path:    Output JSON path for active probe lists
            results: Pre-computed results dict (or None to compute)
            dead:    Enable dead-probe elimination
            merge:   Enable adjacent-probe merging
        """
        if results is None:
            results = self.get_results()

        data = self._build_raw_array(results)
        loc_map = self._loc_map
        n_probes = self._n_probes

        active_start = list(range(n_probes))
        active_end = list(range(n_probes))

        if dead:
            active_start, active_end = self._clear_dead(data)
            n_dead_start = n_probes - len(active_start)
            n_dead_end = n_probes - len(active_end)
            print(f"[Sanitizer] Dead elimination: removed {n_dead_start} start / "
                  f"{n_dead_end} end probes")

        if merge:
            connections = self._get_connections(data)
            merge_list = self._get_merge_list(connections, loc_map)
            eliminated_start = {c[1] for c in merge_list}
            eliminated_end = {c[0] for c in merge_list}
            active_start = [x for x in active_start if x not in eliminated_start]
            active_end = [x for x in active_end if x not in eliminated_end]
            print(f"[Sanitizer] Merge: {len(merge_list)} probe pairs merged")

        json_data = {
            'active_start': {'list': active_start},
            'active_end': {'list': active_end}
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
        print(f"[Sanitizer] Active probes saved: {path}  "
              f"({len(active_start)} start / {len(active_end)} end)")


