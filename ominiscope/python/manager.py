import os
import json
import re
import shutil
import tempfile
import time
import atexit
import configparser
import threading
import queue
from pathlib import Path
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

from insert_ptx import instrument_ptx

DEFAULT_CONFIG = Path(__file__).parent.parent / 'config.ini'
KERNEL_NAME_PATTERN = re.compile(r'\.visible\s+\.entry\s+(\w+)\s*\(')

class ominiscopeManager:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read(DEFAULT_CONFIG)
        self.t_start  = int(self.config['ominiscope']['t_start'])
        self.t_end    = int(self.config['ominiscope']['t_end'])
        self.l_buffer = int(self.config['ominiscope']['l_buffer'])
        self.n_thread = self.t_end - self.t_start + 1
        self.mode = self.config['ominiscope'].get('mode', 'block')
        self.target_file = int(self.config['ominiscope'].get('target_file', 1))
        self.target_kernel = self.config['ominiscope'].get('target_kernel', '').strip() or None
        
        self.loc_map_path = Path(__file__).parent.parent / self.config['analysis']['analysis_dir'] / self.config['analysis']['map']
        self.profiler_buffer = torch.zeros(self.n_thread * self.l_buffer * 4, dtype=torch.int32, device="cuda")
        self.raw_data = None

        self._original_make_cubin = None
        self._original_jit_run = None
        self._loc_map: Dict[int, int] = {}
        self._n_probes: int = 0
        self._active: bool = False
        self._tmp_cache_dir: Optional[str] = None
        self._orig_cache_dir_env: Optional[str] = None
        self._atexit_registered: bool = False
        self._setup_cache_redirect()
        self._activate_hooks()

    def _setup_cache_redirect(self):
        if self._tmp_cache_dir is not None:
            return
        self._orig_cache_dir_env = os.environ.get('TRITON_CACHE_DIR')
        self._tmp_cache_dir = tempfile.mkdtemp(prefix='ominiscope_cache_')
        os.environ['TRITON_CACHE_DIR'] = self._tmp_cache_dir
        if not self._atexit_registered:
            atexit.register(self.close)
            self._atexit_registered = True
        print(f"[Ominiscope] Triton cache redirected -> {self._tmp_cache_dir}")

    def close(self):
        if self._orig_cache_dir_env is None:
            os.environ.pop('TRITON_CACHE_DIR', None)
        else:
            os.environ['TRITON_CACHE_DIR'] = self._orig_cache_dir_env

        if self._tmp_cache_dir and os.path.exists(self._tmp_cache_dir):
            shutil.rmtree(self._tmp_cache_dir, ignore_errors=True)
            print(f"[Ominiscope] Triton cache cleaned: {self._tmp_cache_dir}")
        self._tmp_cache_dir = None
        self._orig_cache_dir_env = None

    def _activate_hooks(self):
        if self._active:
            return

        from triton.backends.nvidia.compiler import CUDABackend
        self._original_make_cubin = CUDABackend.make_cubin

        instrument = self
        original = self._original_make_cubin

        def hooked(backend_self, src, metadata, opt, capability):
            return instrument._hooked_make_cubin(
                original, backend_self, src, metadata, opt, capability
            )

        CUDABackend.make_cubin = hooked
        self._ensure_jit_run_patch()
        self._active = True

    def _find_kernel_name(self, ptx: str) -> Optional[str]:
        for line in ptx.splitlines():
            m = re.match(KERNEL_NAME_PATTERN, line.rstrip())
            if m:
                return m.group(1)
        return None

    def _should_inject_for_kernel(self, kernel_name: Optional[str]) -> bool:
        if not self._active:
            return False
        if kernel_name is None:
            return False
        if self.target_kernel is not None and kernel_name != self.target_kernel:
            return False
        return True

    def _ensure_jit_run_patch(self):
        from triton.runtime.jit import JITFunction
        if self._original_jit_run is not None:
            return

        instrument = self
        self._original_jit_run = JITFunction.run

        def hooked(jit_self, *args, grid, warmup, **kwargs):
            from triton.runtime.jit import compute_cache_key, driver, knobs

            kwargs["debug"] = kwargs.get("debug", jit_self.debug) or knobs.runtime.debug
            kwargs["instrumentation_mode"] = knobs.compilation.instrumentation_mode

            device = driver.active.get_current_device()
            stream = driver.active.get_current_stream(device)

            for hook in jit_self.pre_run_hooks:
                hook(*args, **kwargs)

            kernel_cache, kernel_key_cache, target, backend, binder = jit_self.device_caches[device]
            bound_args, specialization, options = binder(*args, **kwargs)

            key = compute_cache_key(kernel_key_cache, specialization, options)
            kernel = kernel_cache.get(key, None)

            if kernel is None:
                options, signature, constexprs, attrs = jit_self._pack_args(
                    backend, kwargs, bound_args, specialization, options
                )
                kernel_name = getattr(getattr(jit_self, "fn", None), "__name__", None)
                if instrument._should_inject_for_kernel(kernel_name):
                    if hasattr(jit_self, "arg_names"):
                        arg_names = list(jit_self.arg_names)
                        if "profiler_buffer" not in arg_names:
                            arg_names.append("profiler_buffer")
                            jit_self.arg_names = arg_names
                    signature["profiler_buffer"] = "*i32"
                kernel = jit_self._do_compile(key, signature, device, constexprs, options, attrs, warmup)
                if kernel is None:
                    return None

            not_present = object()
            for (name, _), (val, globals_dict) in jit_self.used_global_vals.items():
                if (new_val := globals_dict.get(name, not_present)) != val:
                    raise RuntimeError(
                        f"Global variable {name} has changed since we compiled this kernel, from {val} to {new_val}"
                    )

            if not warmup:
                assert grid is not None
                if callable(grid):
                    grid = grid(bound_args)
                grid_size = len(grid)
                grid_0 = grid[0]
                grid_1 = grid[1] if grid_size > 1 else 1
                grid_2 = grid[2] if grid_size > 2 else 1
                if hasattr(kernel, "result"):
                    kernel = kernel.result()

                kernel_name = getattr(getattr(jit_self, "fn", None), "__name__", None)
                launch_args = tuple(bound_args.values())
                if instrument._should_inject_for_kernel(kernel_name):
                    launch_args = launch_args + (instrument.profiler_buffer,)

                launch_metadata = kernel.launch_metadata(grid, stream, *launch_args)
                kernel.run(
                    grid_0,
                    grid_1,
                    grid_2,
                    stream,
                    kernel.function,
                    kernel.packed_metadata,
                    launch_metadata,
                    knobs.runtime.launch_enter_hook,
                    knobs.runtime.launch_exit_hook,
                    *launch_args,
                )
            return kernel

        JITFunction.run = hooked

    def _hooked_make_cubin(self, original_fn, self_backend, src, metadata, opt, capability):
        kernel_name = self._find_kernel_name(src)
        if self.target_kernel is not None and kernel_name is not None and kernel_name != self.target_kernel:
            return original_fn(self_backend, src, metadata, opt, capability)

        if self._active:
            try:
                dump = self.config.getboolean('output', 'dump')
                if dump:
                    output_dir = Path(__file__).parent.parent / self.config['output']['output_dir']
                    output_dir.mkdir(parents=True, exist_ok=True)
                    orig_path = output_dir / "original.ptx"
                    with open(orig_path, 'w') as f:
                        f.write(src)

                elimination = self.config.getboolean('elimination', 'enable')
                active_list = None
                if elimination:
                    analysis_dir = Path(__file__).parent.parent / self.config['analysis']['analysis_dir']
                    active_list = str(analysis_dir / self.config['analysis']['active'])

                modified_ptx, loc_map, n_probes = instrument_ptx(
                    src, mode=self.mode, t_start=self.t_start, t_end=self.t_end,
                    target_file=self.target_file, active_list=active_list
                )
                self._loc_map = loc_map
                self._n_probes = n_probes

                if dump:
                    output_dir = Path(__file__).parent.parent / self.config['output']['output_dir']
                    inst_path = output_dir / "instrumented.ptx"
                    with open(inst_path, 'w') as f:
                        f.write(modified_ptx)

                analysis_dir = Path(__file__).parent.parent / self.config['analysis']['analysis_dir']
                analysis_dir.mkdir(parents=True, exist_ok=True)
                loc_path = analysis_dir / self.config['analysis']['map']
                with open(loc_path, 'w') as f:
                    json.dump(loc_map, f, indent=2)

                src = modified_ptx
            except Exception as e:
                print(f"[Ominiscope] Instrumentation failed: {e}")
                import traceback
                traceback.print_exc()
                print("[Ominiscope] Falling back to original PTX")

        return original_fn(self_backend, src, metadata, opt, capability)

    def export_raw(self):
        raw_csv_path = Path(__file__).parent.parent / self.config['output']['output_dir'] / f'{self.config["output"]["raw_base"]}.csv'
        raw_npy_path = Path(__file__).parent.parent / self.config['output']['output_dir'] / f'{self.config["output"]["raw_base"]}.npy'
        self.profiler_buffer = self.profiler_buffer.cpu().numpy()

        rows = len(self.profiler_buffer) // (self.n_thread * 4)
        profiler_buffer_list = []

        export_raw_start = time.perf_counter()
        profiler_buffer_reshaped = self.profiler_buffer.view(np.uint32).reshape(rows, self.n_thread * 4)

        for row_idx in range(rows):
            row_data = []
            for thread_idx in range(self.n_thread):
                start_lo = int(profiler_buffer_reshaped[row_idx, thread_idx * 4])
                start_hi = int(profiler_buffer_reshaped[row_idx, thread_idx * 4 + 1])
                end_lo   = int(profiler_buffer_reshaped[row_idx, thread_idx * 4 + 2])
                end_hi   = int(profiler_buffer_reshaped[row_idx, thread_idx * 4 + 3])
                start_idx  = start_hi >> 16
                end_idx    = end_hi >> 16
                start_time = ((start_hi & 0xFFFF) << 32) | start_lo
                end_time   = ((end_hi   & 0xFFFF) << 32) | end_lo
                row_data.extend([start_idx, end_idx, start_time, end_time])
            profiler_buffer_list.append(row_data)

        profiler_buffer_list = [row for row in profiler_buffer_list if any(value != 0 for value in row)]
        self.raw_data = np.array(profiler_buffer_list)
        
        raw_npy_path.parent.mkdir(parents=True, exist_ok=True)
        raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_npy_path.open('wb') as f:
            np.save(f, self.raw_data)
        with raw_csv_path.open('w', encoding='utf-8') as f:
            np.savetxt(f, self.raw_data, delimiter=',', fmt='%d')

        export_raw_end = time.perf_counter()
        print(f"[Ominiscope] Raw data successfully exported, export time: {export_raw_end - export_raw_start:.4f} seconds")

        t0_start = self.raw_data[:, 2]
        t0_end = self.raw_data[:, 3]
        print(f"[Ominiscope] Total cycles of the first probed thread: {np.sum(t0_end - t0_start)}")

        self.__sanitizer()

        return rows

    def export_trace(self, chunk_size: int = 4096, workers: int = 16, queue_size: int = 16) -> Dict[str, Any]:
        trace_path = Path(__file__).parent.parent / self.config['output']['output_dir'] / self.config['output']['trace']
        with self.loc_map_path.open('r', encoding='utf-8') as f:
            loc_probe_map = json.load(f)

        q: "queue.Queue[object]" = queue.Queue(maxsize=queue_size)

        def worker(start_row: int, end_row: int):
            parts = []
            for row in self.raw_data[start_row:end_row]:
                for i in range(self.n_thread):
                    start_index = int(row[i * 4])
                    end_index   = int(row[i * 4 + 1])
                    start_time  = int(row[i * 4 + 2])
                    end_time    = int(row[i * 4 + 3])

                    if start_time == 0:
                        continue

                    duration = end_time - start_time
                    loc = loc_probe_map.get(str(start_index), None)
                    if loc is None:
                        name = f"Line {start_index}"
                        cat = name
                    else:
                        name = f"Line {loc}"
                        cat = f"Line {loc}"

                    event = {
                        "name": name,
                        "cat": cat,
                        "ph": "X",
                        "ts": start_time,
                        "dur": duration,
                        "pid": 0,
                        "tid": i,
                        "args": {
                            "start_index": start_index,
                            "end_index": end_index
                        }
                    }
                    parts.append(json.dumps(event, ensure_ascii=False))

            if parts:
                q.put(',\n'.join(parts))

        def writer_thread_fn():
            first = True
            with trace_path.open('w', encoding='utf-8') as f:
                f.write('{' + f'"schemaVersion": 1, "traceEvents": [\n')
                while True:
                    item = q.get()
                    if item is None:
                        break
                    if not first:
                        f.write(',\n')
                    f.write(item)
                    first = False
                f.write('\n],')
                f.write(f'"displayTimeUnit": "ns"' + '}')

        export_trace_start = time.perf_counter()
        total_rows = self.raw_data.shape[0]

        writer = threading.Thread(target=writer_thread_fn, daemon=True)
        writer.start()

        ranges = []
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            ranges.append((start, end))

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(worker, s, e) for s, e in ranges]
            for fut in futures:
                fut.result()

        q.put(None)
        writer.join()
        export_trace_end = time.perf_counter()
        print(f"[Ominiscope] Trace successfully exported, export time: {export_trace_end - export_trace_start:.4f} seconds")

        return {"schemaVersion": 1, "traceEvents": "streamed", "displayTimeUnit": "ns"}

    def __sanitizer(self):
        from sanitizer import _clear_dead, _merge
        need_active = self.config.getboolean('elimination', 'dead')
        need_merge  = self.config.getboolean('elimination', 'merge')
        
        active_path = Path(__file__).parent.parent / self.config['analysis']['analysis_dir'] / self.config['analysis']['active']

        with self.loc_map_path.open('r', encoding='utf-8') as f:
            loc_probe_map = json.load(f)

        if not need_active and not need_merge:
            active_start = list(range(len(loc_probe_map)))
            active_end = list(range(len(loc_probe_map)))
        else:
            with active_path.open('r', encoding='utf-8') as f:
                active_list  = json.load(f)
                active_start = active_list['active_start']['list']
                active_end   = active_list['active_end']['list']

        if need_active:
            active_start, active_end = _clear_dead(data=self.raw_data)
            print(f"[Ominiscope] Dead probes eliminated successfully")

        if need_merge:
            active_start, active_end = _merge(
                active_start=active_start,
                active_end=active_end,
                data=self.raw_data,
                loc_probe_map=loc_probe_map
            )
            print(f"[Ominiscope] Probes merged successfully")

        json_data = {
            'active_start': {'list': active_start},
            'active_end': {'list': active_end}
        }
        with active_path.open('w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)

        return len(active_start)
