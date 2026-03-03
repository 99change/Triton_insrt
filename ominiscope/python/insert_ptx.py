import re
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from subBlock import builder

PROBE_DIR = Path(__file__).parent.parent / 'probe'

KERNEL_NAME_PATTERN = re.compile(r'\.visible\s+\.entry\s+(\w+)\s*\(')


def _find_kernel_name(lines: List[str]) -> Optional[str]:
    for line in lines:
        m = re.match(KERNEL_NAME_PATTERN, line.rstrip())
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


def insert_multiple_lines(lines: List[str], ptx_line_num: int, line_list: List[str]) -> int:
    insert_idx = max(0, ptx_line_num - 1)
    normalized = []
    for line in line_list:
        if not line.endswith('\n'):
            normalized.append(line + '\n')
        else:
            normalized.append(line)
    lines[insert_idx:insert_idx] = normalized
    return len(normalized)


def replace_mark(lines: List[str], pattern, patch: str) -> List[str]:
    result = []
    for line in lines:
        new_line = re.sub(pattern, patch, line)
        result.append(new_line)
    return result


def instrument_ptx(
    ptx_str: str,
    mode: str = "block",
    t_start: int = 0,
    t_end: int = 127,
    target_file: int = 1,
    active_list: str = None
) -> Tuple[str, Dict[int, int], int]:
    lines = [line + '\n' for line in ptx_str.split('\n')]

    kernel_name = _find_kernel_name(lines)
    if kernel_name is None:
        raise ValueError("No kernel entry point found in PTX")

    time_param = _find_buffer_param_index(lines, kernel_name)

    head_lines = (PROBE_DIR / 'head.ptx').read_text(encoding='utf-8').splitlines(True)
    entry_lines = (PROBE_DIR / 'entry.ptx').read_text(encoding='utf-8').splitlines(True)
    exit_lines = (PROBE_DIR / 'exit.ptx').read_text(encoding='utf-8').splitlines(True)

    total = t_end - t_start + 1

    head_lines = replace_mark(head_lines, r'KERNEL_NAME', kernel_name)
    head_lines = replace_mark(head_lines, r'PARAM', f"{time_param}")
    head_lines = replace_mark(head_lines, r'START', f"{t_start}")
    head_lines = replace_mark(head_lines, r'END', f"{t_end}")
    head_lines = replace_mark(head_lines, r'TOTAL', f"{total}")

    subBlock_list = builder(lines)

    loc_probe_map: Dict[int, int] = {}
    start_line = subBlock_list[0].entry
    probe_pair_list = []
    i = 0

    if mode == "block":
        probe_pair = [None, None]
        for j, block in enumerate(subBlock_list):
            if block.is_start or block.is_end or block.is_sync or block.file != target_file:
                continue
            loc_probe_map[i] = block.loc

            next_is_sync = (j + 1 < len(subBlock_list) and subBlock_list[j + 1].is_sync)
            prev_is_sync = (j > 0 and subBlock_list[j - 1].is_sync)

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
        probe_pair_list.append([subBlock_list[1].entry, subBlock_list[-2].exit])
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

    offset = 0
    offset += insert_multiple_lines(lines, start_line, head_lines)

    for idx, pair in enumerate(probe_pair_list):
        insert_idx = idx << 16
        if pair[0] is not None:
            entry_insert = replace_mark(entry_lines[:], r'INDEX', f"{insert_idx}")
            entry_insert = replace_mark(entry_insert, r'NO', f"{idx}")
            offset += insert_multiple_lines(lines, pair[0] + offset, entry_insert)
        if pair[1] is not None:
            exit_insert = replace_mark(exit_lines[:], r'INDEX', f"{insert_idx}")
            exit_insert = replace_mark(exit_insert, r'NO', f"{idx}")
            offset += insert_multiple_lines(lines, pair[1] + offset + 1, exit_insert)

    n_probes = i
    modified_ptx = ''.join(lines)

    return modified_ptx, loc_probe_map, n_probes
