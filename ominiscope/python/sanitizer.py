import numpy as np
from typing import Set, Tuple, List

def _merge(active_start: List, active_end: List, data: np.ndarray, loc_probe_map: dict[int, int]) -> List[List[int]]:
    connections = _get_connection(data=data)
    merged_connections = _get_merge_list(connections=connections, loc_probe_map=loc_probe_map)

    eliminated_start = {connection[1] for connection in merged_connections}
    eliminated_end   = {connection[0] for connection in merged_connections}
    active_start     = [x for x in active_start if x not in eliminated_start]
    active_end       = [x for x in active_end if x not in eliminated_end]

    return [active_start, active_end]

def _clear_dead(data: np.ndarray) -> List[List[int]]:
    start_idx = data[:, ::4]
    end_idx = data[:, 1::4]
    active_start = np.unique(start_idx).tolist()
    active_end = np.unique(end_idx).tolist()
    
    return [active_start, active_end]

def _get_connection(data: np.ndarray) -> Set[Tuple[int, int]]:
    start_index = data[:, ::4]
    end_index = data[:, 1::4]

    valid = np.cumprod(start_index[1:] != 0, axis=0).astype(bool)

    from_nodes = end_index[:-1][valid]
    to_nodes = start_index[1:][valid]

    return set(zip(from_nodes.tolist(), to_nodes.tolist()))

def _get_merge_list(connections: Set[Tuple[int, int]], loc_probe_map: dict[int, int]) -> List[Tuple[int, int]]:
    merged_connections = []
    for connection in connections:
        from_index = connection[0]
        to_index   = connection[1]
        is_unique_from = sum(1 for c in connections if c[0] == from_index) == 1
        is_unique_to   = sum(1 for c in connections if c[1] == to_index) == 1
        if is_unique_from and is_unique_to:
            from_loc = loc_probe_map.get(str(from_index), None)
            to_loc = loc_probe_map.get(str(to_index), None)
            if from_loc == to_loc:
                merged_connections.append(connection)

    return merged_connections