"""
Sanitizer: dead-probe elimination and adjacent-probe merging.

Usage:
    python -m tri_ins.sanitizer

Reads config from tri_ins/config.ini, loads raw CSV and loc_map.json,
performs dead elimination and/or merge, outputs active_probes.json.
"""

import os
import json
import configparser
import numpy as np
from typing import Dict, List, Set, Tuple

# Default config path (alongside this module)
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')


def clear_dead(data: np.ndarray) -> Tuple[List[int], List[int]]:
    """Find probes that were actually executed (non-dead)."""
    start_idx = data[:, ::4]
    end_idx = data[:, 1::4]
    active_start = sorted(set(np.unique(start_idx).tolist()))
    active_end = sorted(set(np.unique(end_idx).tolist()))
    return active_start, active_end


def get_connections(data: np.ndarray) -> Set[Tuple[int, int]]:
    """Find (end_idx -> next start_idx) transitions across consecutive probes."""
    start_index = data[:, ::4]
    end_index = data[:, 1::4]
    valid = np.cumprod(start_index[1:] != 0, axis=0).astype(bool)
    from_nodes = end_index[:-1][valid]
    to_nodes = start_index[1:][valid]
    return set(zip(from_nodes.tolist(), to_nodes.tolist()))


def get_merge_list(connections: Set[Tuple[int, int]],
                   loc_map: Dict[int, int]) -> List[Tuple[int, int]]:
    """Find probe pairs that can be merged (unique 1:1 connection, same loc)."""
    merged = []
    for from_idx, to_idx in connections:
        is_unique_from = sum(1 for c in connections if c[0] == from_idx) == 1
        is_unique_to = sum(1 for c in connections if c[1] == to_idx) == 1
        if is_unique_from and is_unique_to:
            from_loc = loc_map.get(from_idx, loc_map.get(str(from_idx), None))
            to_loc = loc_map.get(to_idx, loc_map.get(str(to_idx), None))
            if from_loc is not None and from_loc == to_loc:
                merged.append((from_idx, to_idx))
    return merged


def sanitize(raw_csv: str, loc_map_path: str, output_path: str,
             dead: bool = True, merge: bool = True):
    """
    Perform dead-probe elimination and/or merge on raw CSV data.

    Args:
        raw_csv:      Path to raw.csv (probes x threads*4)
        loc_map_path: Path to loc_map.json
        output_path:  Path to write active_probes.json
        dead:         Enable dead-probe elimination
        merge:        Enable adjacent-probe merging
    """
    # Load raw CSV (skip header row)
    data = np.loadtxt(raw_csv, delimiter=',', dtype=np.int64, skiprows=1)

    with open(loc_map_path, 'r', encoding='utf-8') as f:
        loc_map = json.load(f)
    # Convert string keys to int
    loc_map = {int(k): v for k, v in loc_map.items()}

    n_probes = data.shape[0]
    active_start = list(range(n_probes))
    active_end = list(range(n_probes))

    if dead:
        active_start, active_end = clear_dead(data)
        n_dead_start = n_probes - len(active_start)
        n_dead_end = n_probes - len(active_end)
        print(f"[Sanitizer] Dead elimination: removed {n_dead_start} start / "
              f"{n_dead_end} end probes")

    if merge:
        connections = get_connections(data)
        merge_list = get_merge_list(connections, loc_map)
        eliminated_start = {c[1] for c in merge_list}
        eliminated_end = {c[0] for c in merge_list}
        active_start = [x for x in active_start if x not in eliminated_start]
        active_end = [x for x in active_end if x not in eliminated_end]
        print(f"[Sanitizer] Merge: {len(merge_list)} probe pairs merged")

    json_data = {
        'active_start': {'list': active_start},
        'active_end': {'list': active_end}
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    print(f"[Sanitizer] Active probes saved: {output_path}  "
          f"({len(active_start)} start / {len(active_end)} end)")


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

    dead = san_sec.getboolean('dead', True)
    merge = san_sec.getboolean('merge', True)

    sanitize(raw_csv, loc_map_path, active_list, dead=dead, merge=merge)


if __name__ == '__main__':
    main()
