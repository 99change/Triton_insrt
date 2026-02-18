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
    from triton_instrument import TritonInstrument

    # Add buffer_ptr to kernel signature (last non-constexpr param)
    @triton.jit
    def my_kernel(x_ptr, n, buffer_ptr, BLOCK: tl.constexpr):
        ...

    with TritonInstrument(mode="block") as inst:
        buf = inst.allocate_buffer(n_probes_estimate=64, n_threads=128)
        my_kernel[grid](x, n, buf, BLOCK=128)
        torch.cuda.synchronize()
        results = inst.get_results()
"""

import os
import re
import json
import torch
from typing import List, Dict, Tuple, Optional
from ptx_parser import sub_block, builder

# ---- Paths to PTX probe templates (from Exist_Package) ----
PACKAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Exist_Package')


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


def _find_buffer_param_index(lines: List[str], kernel_name: str) -> int:
    """
    Find the parameter index of buffer_ptr in PTX.

    Triton appends 2 hidden params (global_scratch, profile_scratch) after
    all user params. Our buffer_ptr is the last user param, so it's at
    index = max_param_index - 2.
    """
    param_pattern = re.compile(rf'{re.escape(kernel_name)}_param_(\d+)')
    max_idx = -1
    for line in lines:
        m = param_pattern.search(line)
        if m:
            idx = int(m.group(1))
            max_idx = max(max_idx, idx)
    if max_idx < 2:
        raise ValueError(f"Found only {max_idx+1} params — need at least 3 "
                         f"(user params + global_scratch + profile_scratch)")
    return max_idx - 2


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

    buffer_param = _find_buffer_param_index(lines, kernel_name)

    # Load probe templates from Exist_Package
    head_lines = _load_template('head.ptx')
    entry_lines = _load_template('entry.ptx')
    exit_lines = _load_template('exit.ptx')
    config_lines = _load_template('config.ptx')

    total = t_end - t_start + 1

    # Parameterize head template
    head_lines = _replace_mark(head_lines, r'KERNEL_NAME', kernel_name)
    head_lines = _replace_mark(head_lines, r'PARAM', str(buffer_param))
    head_lines = _replace_mark(head_lines, r'START', str(t_start))
    head_lines = _replace_mark(head_lines, r'END', str(t_end))
    head_lines = _replace_mark(head_lines, r'TOTAL', str(total))

    # Parameterize config template
    config_lines = _replace_mark(config_lines, r'KERNEL_NAME', kernel_name)
    config_lines = _replace_mark(config_lines, r'PARAM', str(buffer_param))

    # Parse PTX into sub-blocks
    sub_block_list = builder(lines)

    print(f"[Instrument] Kernel: {kernel_name}")
    print(f"[Instrument] Buffer param index: {buffer_param}")
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
            if block.is_start or block.is_end or block.is_sync:
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
    The user's kernel must include `buffer_ptr` as its last non-constexpr parameter.

    Example:
        with TritonInstrument(mode="block", t_start=0, t_end=127) as inst:
            kernel[grid](x, y, inst.buffer, BLOCK=128)
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
        self._buffer: Optional[torch.Tensor] = None
        self._active: bool = False
        self._dump_ptx: Optional[str] = None  # Path to dump instrumented PTX

    @property
    def buffer(self) -> Optional[torch.Tensor]:
        """The GPU buffer tensor. Allocate with allocate_buffer() before kernel launch."""
        return self._buffer

    @property
    def loc_map(self) -> Dict[int, int]:
        """Mapping from probe_id to Python source line number."""
        return self._loc_map

    @property
    def n_probes(self) -> int:
        """Number of probes inserted (available after first compilation)."""
        return self._n_probes

    def allocate_buffer(self, n_probes_estimate: int = 128,
                        n_threads: Optional[int] = None) -> torch.Tensor:
        """
        Pre-allocate the output buffer for probe data.

        Buffer layout (per the existing cuTile convention):
            For each probe slot:
                For each thread:
                    [start_idx, end_idx, start_time, end_time]  (4 x int64 = 32 bytes)

        Total size = n_probes * n_threads * 4 * 8 bytes

        Args:
            n_probes_estimate: Estimated number of probes (overallocate is fine)
            n_threads: Number of probed threads (t_end - t_start + 1 by default)

        Returns:
            torch.Tensor on CUDA, pass this as buffer_ptr to the kernel
        """
        if n_threads is None:
            n_threads = self.t_end - self.t_start + 1
        total_elements = n_probes_estimate * n_threads * 4
        self._buffer = torch.zeros(total_elements, dtype=torch.int64, device='cuda')
        print(f"[Instrument] Buffer allocated: {total_elements * 8} bytes "
              f"({n_probes_estimate} probes x {n_threads} threads x 32B)")
        return self._buffer

    def _hooked_make_cubin(self, original_fn, self_backend, src, metadata, opt, capability):
        """Intercept PTX, insert probes, then compile to cubin."""
        if self._active:
            try:
                modified_ptx, loc_map, n_probes = instrument_ptx(
                    src, mode=self.mode, t_start=self.t_start, t_end=self.t_end
                )
                self._loc_map = loc_map
                self._n_probes = n_probes

                print(f"\n[Instrument] ✅ Inserted {n_probes} probes, mode={self.mode}")
                print(f"[Instrument] Thread range: [{self.t_start}, {self.t_end}]")
                print(f"[Instrument] Loc map: {json.dumps(loc_map, indent=2)}")

                if self._dump_ptx:
                    with open(self._dump_ptx, 'w') as f:
                        f.write(modified_ptx)
                    print(f"[Instrument] Dumped instrumented PTX to {self._dump_ptx}")

                src = modified_ptx
            except Exception as e:
                print(f"[Instrument] ❌ Instrumentation failed: {e}")
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
        Parse the buffer tensor and return structured timing data.

        Returns:
            Dict mapping probe_id -> {
                'loc': int,           # Python source line number
                'start_idx': Tensor,  # probe start index per thread
                'end_idx': Tensor,    # probe end index per thread
                'start_time': Tensor, # clock64 at entry
                'end_time': Tensor,   # clock64 at exit
                'duration': Tensor,   # end_time - start_time
            }
        """
        if self._buffer is None:
            raise RuntimeError("No buffer allocated. Call allocate_buffer() first.")

        n_threads = self.t_end - self.t_start + 1
        n_probes = self._n_probes
        if n_probes == 0:
            print("[Instrument] Warning: n_probes=0, no probes were inserted")
            return {}

        # Reshape: [n_probes, n_threads, 4]
        data = self._buffer[:n_probes * n_threads * 4].reshape(n_probes, n_threads, 4).cpu()

        results = {}
        for probe_idx in range(n_probes):
            probe_id = probe_idx + 1
            loc = self._loc_map.get(probe_id, -1)
            probe_data = data[probe_idx]
            duration = probe_data[:, 3] - probe_data[:, 2]
            results[probe_id] = {
                'loc': loc,
                'start_idx': probe_data[:, 0],
                'end_idx': probe_data[:, 1],
                'start_time': probe_data[:, 2],
                'end_time': probe_data[:, 3],
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
