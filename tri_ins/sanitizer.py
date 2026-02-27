import os
import json
import configparser
import numpy as np
from typing import Dict, List, Set, Tuple

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')


def clear_dead(data: np.ndarray) -> Tuple[List[int], List[int]]:
    start_idx = data[:, ::4]
    end_idx = data[:, 1::4]
    active_start = sorted(set(np.unique(start_idx).tolist()))
    active_end = sorted(set(np.unique(end_idx).tolist()))
    return active_start, active_end


def get_connections(data: np.ndarray) -> Set[Tuple[int, int]]:
    start_index = data[:, ::4]
    end_index = data[:, 1::4]
    valid = np.cumprod(start_index[1:] != 0, axis=0).astype(bool)
    from_nodes = end_index[:-1][valid]
    to_nodes = start_index[1:][valid]
    return set(zip(from_nodes.tolist(), to_nodes.tolist()))


def get_merge_list(connections: Set[Tuple[int, int]],
                   loc_map: Dict[int, int]) -> List[Tuple[int, int]]:
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
    data = np.loadtxt(raw_csv, delimiter=',', dtype=np.int64)

    with open(loc_map_path, 'r', encoding='utf-8') as f:
        loc_map = json.load(f)
    loc_map = {int(k): v for k, v in loc_map.items()}

    probe_site_ids = sorted(loc_map.keys())
    active_start = list(probe_site_ids)
    active_end = list(probe_site_ids)

    if dead:
        active_start, active_end = clear_dead(data)
        # clear_dead 返回 raw data 中实际出现过的 probe site ID，
        # 需再与 loc_map 取交集，过滤掉不在探针表里的异常值
        active_start = sorted(set(active_start) & set(probe_site_ids))
        active_end   = sorted(set(active_end)   & set(probe_site_ids))
        n_dead_start = len(probe_site_ids) - len(active_start)
        n_dead_end   = len(probe_site_ids) - len(active_end)
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
    cfg = configparser.ConfigParser()
    cfg.read(DEFAULT_CONFIG)

    out_sec = cfg['output']
    ana_sec = cfg['analysis']
    san_sec = cfg['sanitizer']

    output_dir = out_sec.get('output_dir', 'output')
    analysis_dir = ana_sec.get('analysis_dir', 'analysis')
    raw_csv = os.path.join(output_dir, out_sec.get('raw_csv', 'raw.csv'))
    loc_map_path = os.path.join(analysis_dir, ana_sec.get('loc_map', 'loc_map.json'))
    active_list = os.path.join(analysis_dir, ana_sec.get('active_list', 'active_probes.json'))

    dead = san_sec.getboolean('dead', True)
    merge = san_sec.getboolean('merge', True)

    sanitize(raw_csv, loc_map_path, active_list, dead=dead, merge=merge)


if __name__ == '__main__':
    main()
