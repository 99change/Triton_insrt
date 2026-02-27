import os
import re
import json
import shutil
import tempfile
import configparser
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from .ptx_parser import sub_block, builder

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')
PROBE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'probe')


def _load_template(name: str) -> List[str]:
    path = os.path.join(PROBE_DIR, name)
    with open(path, 'r', encoding='utf-8') as f:
        return f.readlines()


def _replace_mark(lines: List[str], pattern: str, replacement: str) -> List[str]:
    return [re.sub(pattern, replacement, line) for line in lines]


def _insert_lines(lines: List[str], line_num_1based: int, new_lines: List[str]) -> int:
    idx = line_num_1based - 1
    for i, new_line in enumerate(new_lines):
        if not new_line.endswith('\n'):
            new_line += '\n'
        lines.insert(idx + i, new_line)
    return len(new_lines)


def _find_kernel_name(lines: List[str]) -> Optional[str]:
    for line in lines:
        m = re.match(r'\s*\.visible\s+\.entry\s+(\w+)\s*\(', line)
        if m:
            return m.group(1)
    return None


def _find_buffer_param_index(lines: List[str], kernel_name: str) -> int:
    param_pattern = re.compile(rf'{re.escape(kernel_name)}_param_(\d+)')
    max_idx = -1
    for line in lines:
        m = param_pattern.search(line)
        if m:
            idx = int(m.group(1))
            max_idx = max(max_idx, idx)
    if max_idx < 2:
        raise ValueError(f"Found only {max_idx+1} params — need at least 3 "
                         f"(user params + time_buffer + 2 hidden)")
    return max_idx - 2


def instrument_ptx(ptx_str: str,
                   mode: str = "block",
                   t_start: int = 0,
                   t_end: int = 127,
                   target_file: int = 1,
                   active_list: str = None) -> Tuple[str, Dict[int, int], int]:
    lines = [line + '\n' for line in ptx_str.split('\n')]

    kernel_name = _find_kernel_name(lines)
    if kernel_name is None:
        raise ValueError("No kernel entry point found in PTX")

    time_param = _find_buffer_param_index(lines, kernel_name)

    head_lines = _load_template('head.ptx')
    entry_lines = _load_template('entry.ptx')
    exit_lines = _load_template('exit.ptx')

    total = t_end - t_start + 1

    head_lines = _replace_mark(head_lines, r'KERNEL_NAME', kernel_name)
    head_lines = _replace_mark(head_lines, r'PARAM', f"{time_param}")
    head_lines = _replace_mark(head_lines, r'START', f"{t_start}")
    head_lines = _replace_mark(head_lines, r'END', f"{t_end}")
    head_lines = _replace_mark(head_lines, r'TOTAL', f"{total}")

    sub_block_list = builder(lines)

    print(f"[Instrument] Kernel: {kernel_name}")
    print(f"[Instrument] time_param={time_param}")
    print(f"[Instrument] Sub-blocks found: {len(sub_block_list)}  (target_file={target_file})")
    file_indices = sorted({b.file for b in sub_block_list})
    print(f"[Instrument] Source file indices seen in .loc: {file_indices}")
    for blk in sub_block_list:
        blk.print_block()

    loc_probe_map: Dict[int, int] = {}
    start_line = sub_block_list[0].entry
    probe_pair_list = []
    i = 0

    if mode == "block":
        probe_pair = [None, None]
        for j, block in enumerate(sub_block_list):
            if block.is_start or block.is_end or block.is_sync or block.file != target_file:
                continue
            loc_probe_map[i] = block.loc

            next_is_sync = (j + 1 < len(sub_block_list) and sub_block_list[j + 1].is_sync)
            prev_is_sync = (j > 0 and sub_block_list[j - 1].is_sync)

            if next_is_sync:
                probe_pair[0] = block.entry
            elif prev_is_sync:
                probe_pair[1] = block.exit
                probe_pair_list.append(probe_pair)
                probe_pair = [None, None]
                i += 1
            else:
                probe_pair[0] = block.entry
                probe_pair[1] = block.exit
                probe_pair_list.append(probe_pair)
                probe_pair = [None, None]
                i += 1
    elif mode == "entire":
        probe_pair_list.append([sub_block_list[1].entry, sub_block_list[-2].exit])
        loc_probe_map[i] = -1
        i += 1
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if active_list is not None:
        with open(active_list, 'r', encoding='utf-8') as f:
            data = json.load(f)
        active_start = data['active_start']['list']
        active_end = data['active_end']['list']
        for idx, pair in enumerate(probe_pair_list):
            new_pair = [None, None]
            if idx in active_start:
                new_pair[0] = pair[0]
            if idx in active_end:
                new_pair[1] = pair[1]
            probe_pair_list[idx] = new_pair
        print(f"[Instrument] Elimination applied from {active_list}")

    offset = 0
    offset += _insert_lines(lines, start_line, head_lines)

    for idx, pair in enumerate(probe_pair_list):
        insert_idx = idx << 16
        if pair[0] is not None:
            entry_insert = _replace_mark(entry_lines[:], r'INDEX', f"{insert_idx}")
            entry_insert = _replace_mark(entry_insert, r'NO', f"{idx}")
            offset += _insert_lines(lines, pair[0] + offset, entry_insert)
        if pair[1] is not None:
            exit_insert = _replace_mark(exit_lines, r'INDEX', f"{insert_idx}")
            exit_insert = _replace_mark(exit_insert, r'NO', f"{idx}")
            offset += _insert_lines(lines, pair[1] + offset + 1, exit_insert)

    n_probes = i
    modified_ptx = ''.join(lines)

    return modified_ptx, loc_probe_map, n_probes


class TritonInstrument:
    def __init__(self, mode: str = "block", t_start: int = 0, t_end: int = 127,
                 target_file: int = 1,
                 output_dir: Optional[str] = "output",
                 analysis_dir: Optional[str] = "analysis",
                 active_list: Optional[str] = None,
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
        self.analysis_dir = analysis_dir
        self.active_list = active_list
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
        self._active: bool = False
        self._tmp_cache_dir: Optional[str] = None
        self._orig_cache_dir_env: Optional[str] = None

    @classmethod
    def from_config(cls, config_path: str = None) -> 'TritonInstrument':
        if config_path is None:
            config_path = DEFAULT_CONFIG
        cfg = configparser.ConfigParser()
        cfg.read(config_path)

        inst_sec = cfg['instrument']
        out_sec = cfg['output']
        ana_sec = cfg['analysis']
        san_sec = cfg['sanitizer'] if 'sanitizer' in cfg else {}

        elimination = san_sec.getboolean('enable', False) if hasattr(san_sec, 'getboolean') else False
        active_list = None
        if elimination:
            analysis_dir = ana_sec.get('analysis_dir', 'analysis')
            active_list = os.path.join(analysis_dir, ana_sec.get('active_list', 'active_probes.json'))

        return cls(
            mode=inst_sec.get('mode', 'block'),
            t_start=inst_sec.getint('t_start', 0),
            t_end=inst_sec.getint('t_end', 127),
            target_file=inst_sec.getint('target_file', 1),
            n_probes_estimate=inst_sec.getint('n_probes_estimate', 256),
            output_dir=out_sec.get('output_dir', 'output'),
            analysis_dir=ana_sec.get('analysis_dir', 'analysis'),
            active_list=active_list,
            raw_csv=out_sec.get('raw_csv', 'raw.csv'),
            trace_json=out_sec.get('trace_json', 'trace.json'),
            loc_map_json=ana_sec.get('loc_map', 'loc_map.json'),
            dump_ptx=out_sec.getboolean('dump_ptx', True),
        )

    @property
    def time_buffer(self) -> Optional[torch.Tensor]:
        return self._time_buffer

    @property
    def loc_map(self) -> Dict[int, int]:
        return self._loc_map

    @property
    def n_probes(self) -> int:
        return self._n_probes

    def allocate_buffer(self, n_probes_estimate: int = 128,
                        n_threads: Optional[int] = None) -> torch.Tensor:
        if n_threads is None:
            n_threads = self.t_end - self.t_start + 1
        self._n_threads = n_threads
        n_elem = n_probes_estimate * n_threads * 4
        self._time_buffer = torch.zeros(n_elem, dtype=torch.int32, device='cuda')
        print(f"[Instrument] time_buffer: {n_elem * 4} bytes (u32 x4)  "
              f"({n_probes_estimate} slots x {n_threads} threads)")
        return self._time_buffer

    def _hooked_make_cubin(self, original_fn, self_backend, src, metadata, opt, capability):
        if self._active:
            try:
                if self.output_dir and self.dump_ptx:
                    os.makedirs(self.output_dir, exist_ok=True)
                    orig_path = os.path.join(self.output_dir, "original.ptx")
                    with open(orig_path, 'w') as f:
                        f.write(src)
                    print(f"[Instrument] Saved original PTX  -> {orig_path}")

                modified_ptx, loc_map, n_probes = instrument_ptx(
                    src, mode=self.mode, t_start=self.t_start, t_end=self.t_end,
                    target_file=self.target_file, active_list=self.active_list
                )
                self._loc_map = loc_map
                self._n_probes = n_probes

                print(f"\n[Instrument] Inserted {n_probes} probes, mode={self.mode}")
                print(f"[Instrument] Thread range: [{self.t_start}, {self.t_end}]")
                print(f"[Instrument] Loc map: {json.dumps(loc_map, indent=2)}")

                if self.output_dir and self.dump_ptx:
                    inst_path = os.path.join(self.output_dir, "instrumented.ptx")
                    with open(inst_path, 'w') as f:
                        f.write(modified_ptx)
                    print(f"[Instrument] Saved instrumented PTX -> {inst_path}")

                if self.analysis_dir:
                    os.makedirs(self.analysis_dir, exist_ok=True)
                    loc_path = os.path.join(self.analysis_dir, self.loc_map_json)
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
        self._orig_cache_dir_env = os.environ.get('TRITON_CACHE_DIR')
        self._tmp_cache_dir = tempfile.mkdtemp(prefix='triton_inst_cache_')
        os.environ['TRITON_CACHE_DIR'] = self._tmp_cache_dir
        print(f"[Instrument] Triton cache redirected -> {self._tmp_cache_dir}")

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
        from triton.backends.nvidia.compiler import CUDABackend
        CUDABackend.make_cubin = self._original_make_cubin
        self._active = False

        if self._orig_cache_dir_env is None:
            os.environ.pop('TRITON_CACHE_DIR', None)
        else:
            os.environ['TRITON_CACHE_DIR'] = self._orig_cache_dir_env

        if self._tmp_cache_dir and os.path.exists(self._tmp_cache_dir):
            shutil.rmtree(self._tmp_cache_dir, ignore_errors=True)
            print(f"[Instrument] Triton cache cleaned: {self._tmp_cache_dir}")
        self._tmp_cache_dir = None
        self._orig_cache_dir_env = None

    def get_results(self, export_path: str = None) -> Dict[int, dict]:
        if self._time_buffer is None:
            raise RuntimeError("No buffers allocated. Call allocate_buffer() first.")

        n_threads = self._n_threads or (self.t_end - self.t_start + 1)
        n_probes = self._n_probes
        if n_probes == 0:
            print("[Instrument] Warning: n_probes=0, no probes were inserted")
            return {}

        n_slots = self.n_probes_estimate
        n_elem_expected = n_slots * n_threads * 4
        buf_np = self._time_buffer[:n_elem_expected].cpu().numpy().view(np.uint32)
        buf_4d = buf_np.reshape(n_slots, n_threads, 4)

        written_mask = (buf_4d != 0).any(axis=(1, 2))
        last_written = int(np.flatnonzero(written_mask)[-1]) + 1 if written_mask.any() else 0
        buf_4d = buf_4d[:last_written]
        n_actual = last_written
        print(f"[Instrument] Static probe sites: {n_probes},  actual buffer slots used: {n_actual}")

        start_lo = buf_4d[:, :, 0]
        start_hi = buf_4d[:, :, 1]
        end_lo   = buf_4d[:, :, 2]
        end_hi   = buf_4d[:, :, 3]

        start_time_64 = ((start_hi & 0xFFFF).astype(np.uint64) << 32) | start_lo.astype(np.uint64)
        end_time_64   = ((end_hi   & 0xFFFF).astype(np.uint64) << 32) | end_lo.astype(np.uint64)
        probe_idx_arr = (start_hi >> 16).astype(np.int64)

        buffer_list_raw = []
        for row_idx in range(n_actual):
            row_data = []
            for thread_idx in range(n_threads):
                pix = int(probe_idx_arr[row_idx, thread_idx])
                st  = int(start_time_64[row_idx, thread_idx])
                et  = int(end_time_64[row_idx, thread_idx])
                row_data.extend([pix, pix, st, et])
            buffer_list_raw.append(row_data)

        buffer_2d_raw = np.array(buffer_list_raw, dtype=np.int64)

        if export_path is not None:
            np.save(export_path + ".npy", buffer_2d_raw)
            np.savetxt(export_path + ".csv", buffer_2d_raw, delimiter=',', fmt='%d')
            print(f"[Instrument] Saved: {export_path}.npy  {export_path}.csv  "
                  f"({n_actual} slots x {n_threads} threads)")

        results = {}
        for slot_idx in range(n_actual):
            row = buffer_2d_raw[slot_idx]
            start_idx_t  = torch.tensor([row[t * 4]     for t in range(n_threads)], dtype=torch.int64)
            end_idx_t    = torch.tensor([row[t * 4 + 1] for t in range(n_threads)], dtype=torch.int64)
            start_time_t = torch.tensor([row[t * 4 + 2] for t in range(n_threads)], dtype=torch.int64)
            end_time_t   = torch.tensor([row[t * 4 + 3] for t in range(n_threads)], dtype=torch.int64)
            duration = end_time_t - start_time_t
            probe_site = int(start_idx_t[0].item())
            results[slot_idx] = {
                'probe_site':  probe_site,
                'loc':         self._loc_map.get(probe_site, -1),
                'start_idx':   start_idx_t,
                'end_idx':     end_idx_t,
                'start_time':  start_time_t,
                'end_time':    end_time_t,
                'duration':    duration,
                'mean_cycles': duration.float().mean().item(),
            }

        return results

    def print_summary(self, results: Optional[Dict] = None):
        if results is None:
            results = self.get_results()

        print("\n" + "=" * 70)
        print("Triton Kernel Instrumentation Results")
        print("=" * 70)
        print(f"{'Slot':>6} {'Site':>6} {'Loc':>6} {'Mean Cycles':>14} {'Min':>12} {'Max':>12}")
        print("-" * 70)
        for slot_id, data in sorted(results.items()):
            dur = data['duration']
            nonzero = dur[dur > 0]
            if len(nonzero) > 0:
                mean = nonzero.float().mean().item()
                mn = nonzero.min().item()
                mx = nonzero.max().item()
            else:
                mean = mn = mx = 0
            print(f"{slot_id:>6} {data['probe_site']:>6} {data['loc']:>6} {mean:>14.1f} {mn:>12} {mx:>12}")
        print("=" * 70)

    def export_chrome_trace(self, path: str, results: Optional[Dict] = None,
                            chunk_size: int = 4096, workers: int = 16,
                            queue_size: int = 16):
        if results is None:
            results = self.get_results()
        n_threads = self._n_threads or (self.t_end - self.t_start + 1)
        loc_map = self._loc_map

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
                f.write('{"displayTimeUnit": "ns", "traceEvents": [\n')
                while True:
                    item = q.get()
                    if item is None:
                        break
                    if not first:
                        f.write(',\n')
                    f.write(item)
                    first = False
                f.write('\n]}')

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