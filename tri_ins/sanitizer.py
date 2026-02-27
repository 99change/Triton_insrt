"""
Sanitizer: dead-probe elimination and adjacent-probe merging.

Usage:
    python -m tri_ins.sanitizer

Reads config from tri_ins/config.ini, loads raw CSV and loc_map.json,
performs dead elimination and/or merge, outputs active_probes.json
and sanitized Chrome trace JSON.
"""

import os
import json
import queue
import threading
import configparser
import numpy as np
from typing import Dict, List, Set, Tuple
from concurrent.futures import ThreadPoolExecutor

# Default config path (alongside this module)
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')


def clear_dead(data: np.ndarray) -> List[List[int]]:
    """Find probes that were actually executed (non-dead)."""
    start_idx = data[:, ::4]
    end_idx = data[:, 1::4]
    active_start = np.unique(start_idx).tolist()
    active_end = np.unique(end_idx).tolist()
    return [active_start, active_end]


def _get_connections(data: np.ndarray) -> Set[Tuple[int, int]]:
    """Find (end_idx -> next start_idx) transitions across consecutive probes."""
    start_index = data[:, ::4]
    end_index = data[:, 1::4]
    valid = np.cumprod(start_index[1:] != 0, axis=0).astype(bool)
    from_nodes = end_index[:-1][valid]
    to_nodes = start_index[1:][valid]
    return set(zip(from_nodes.tolist(), to_nodes.tolist()))


def _get_merge_list(connections: Set[Tuple[int, int]],
                    loc_map: Dict[str, int]) -> List[Tuple[int, int]]:
    """Find probe pairs that can be merged (unique 1:1 connection, same loc)."""
    merged = []
    for connection in connections:
        from_index = connection[0]
        to_index = connection[1]
        is_unique_from = sum(1 for c in connections if c[0] == from_index) == 1
        is_unique_to = sum(1 for c in connections if c[1] == to_index) == 1
        if is_unique_from and is_unique_to:
            from_loc = loc_map.get(str(from_index), None)
            to_loc = loc_map.get(str(to_index), None)
            if from_loc == to_loc:
                merged.append(connection)
    return merged


def merge(active_start: List[int], active_end: List[int],
          data: np.ndarray, loc_map: Dict[str, int]) -> List[List[int]]:
    """Merge adjacent 1:1-connected probes on the same source line."""
    connections = _get_connections(data=data)
    merge_list = _get_merge_list(connections=connections, loc_map=loc_map)

    eliminated_start = {c[1] for c in merge_list}
    eliminated_end = {c[0] for c in merge_list}
    active_start = [x for x in active_start if x not in eliminated_start]
    active_end = [x for x in active_end if x not in eliminated_end]

    return [active_start, active_end]


def export_chrome_trace(path: str, data: np.ndarray,
                        active_start: List[int], active_end: List[int],
                        loc_map: Dict[str, int],
                        chunk_size: int = 4096, workers: int = 16,
                        queue_size: int = 16):
    """
    Export sanitized probe data as Chrome Trace Event JSON.

    Pairs active_start[i] with active_end[i] positionally:
      - ts  = start_time from the start-probe row
      - dur = end_time from the end-probe row  -  start_time
    """
    n_threads = data.shape[1] // 4
    n_pairs = min(len(active_start), len(active_end))

    if n_pairs == 0:
        with open(path, 'w') as f:
            json.dump({"schemaVersion": 1, "traceEvents": [],
                       "displayTimeUnit": "ns"}, f)
        print(f"[Sanitizer] Chrome trace saved: {path}  (empty)")
        return

    q: "queue.Queue[object]" = queue.Queue(maxsize=queue_size)

    def worker(pair_start: int, pair_end: int):
        parts = []
        for p in range(pair_start, pair_end):
            s_row = data[active_start[p]]
            e_row = data[active_end[p]]
            for t in range(n_threads):
                start_idx  = int(s_row[t * 4])
                end_idx    = int(e_row[t * 4 + 1])
                start_time = int(s_row[t * 4 + 2])
                end_time   = int(e_row[t * 4 + 3])

                if start_time == 0:
                    continue

                duration = end_time - start_time
                loc = loc_map.get(str(start_idx), None)
                name = f"Line {loc}" if loc is not None else f"Probe {start_idx}"

                event = {
                    "name": name,
                    "cat": name,
                    "ph": "X",
                    "ts": start_time,
                    "dur": duration,
                    "pid": 0,
                    "tid": t,
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

    writer = threading.Thread(target=writer_fn, daemon=True)
    writer.start()

    ranges = [(s, min(s + chunk_size, n_pairs))
              for s in range(0, n_pairs, chunk_size)]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker, s, e) for s, e in ranges]
        for fut in futures:
            fut.result()

    q.put(None)
    writer.join()

    print(f"[Sanitizer] Chrome trace saved: {path}  "
          f"({n_pairs} probe pairs x {n_threads} threads)")


def sanitize(raw_csv: str, loc_map_path: str, output_path: str,
             trace_path: str = None,
             dead: bool = True, need_merge: bool = True):
    """
    Perform dead-probe elimination and/or merge on raw CSV data.

    Args:
        raw_csv:      Path to raw.csv (probes x threads*4)
        loc_map_path: Path to loc_map.json
        output_path:  Path to write active_probes.json
        trace_path:   Path to write sanitized Chrome trace JSON (optional)
        dead:         Enable dead-probe elimination
        need_merge:   Enable adjacent-probe merging
    """
    # Load raw CSV (skip header row)
    data = np.loadtxt(raw_csv, delimiter=',', dtype=np.int64, skiprows=1)

    with open(loc_map_path, 'r', encoding='utf-8') as f:
        loc_map = json.load(f)
    # Keep loc_map keys as strings (matching original design)

    active_start = list(range(len(loc_map)))
    active_end = list(range(len(loc_map)))

    if dead:
        active_start, active_end = clear_dead(data=data)
        n_dead_start = len(loc_map) - len(active_start)
        n_dead_end = len(loc_map) - len(active_end)
        print(f"[Sanitizer] Dead elimination: removed {n_dead_start} start / "
              f"{n_dead_end} end probes")

    if need_merge:
        active_start, active_end = merge(
            active_start=active_start,
            active_end=active_end,
            data=data,
            loc_map=loc_map
        )
        print(f"[Sanitizer] Merge complete")

    json_data = {
        'active_start': {'list': active_start},
        'active_end': {'list': active_end}
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    print(f"[Sanitizer] Active probes saved: {output_path}  "
          f"({len(active_start)} start / {len(active_end)} end)")

    # Export sanitized Chrome trace
    if trace_path:
        export_chrome_trace(trace_path, data, active_start, active_end, loc_map)


def main():
    """Run sanitizer using settings from config.ini."""
    cfg = configparser.ConfigParser()
    cfg.read(DEFAULT_CONFIG)

    out_sec = cfg['output']
    san_sec = cfg['sanitizer']

    output_dir = out_sec.get('output_dir', 'output')
    raw_csv = os.path.join(output_dir, out_sec.get('raw_csv', 'raw.csv'))
    loc_map_path = os.path.join(output_dir, out_sec.get('loc_map', 'loc_map.json'))
    active_list = os.path.join(output_dir, san_sec.get('active_list', 'active_probes.json'))

    trace_name = san_sec.get('sanitized_trace', 'sanitized_trace.json')
    trace_path = os.path.join(output_dir, trace_name) if trace_name else None

    dead = san_sec.getboolean('dead', True)
    need_merge = san_sec.getboolean('merge', True)

    sanitize(raw_csv, loc_map_path, active_list, trace_path=trace_path,
             dead=dead, need_merge=need_merge)


if __name__ == '__main__':
    main()
