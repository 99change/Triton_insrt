import re
import json
import sys
import shutil
import argparse
import configparser
from typing import List
from sub_block import sub_block, builder

KERNEL_NAME = re.compile(r'\.visible\s+\.entry\s+(\w+)\s*\(')
def is_right_kernel(lines: List[str], name: str) -> bool:
    for line in lines:
        line = line.rstrip()

        kernel_name_match = re.match(KERNEL_NAME, line)
        if kernel_name_match:
            return name == kernel_name_match.group(1)

def insert_a_line(ptx_file: str, ptx_line_num: int, line_to_insert: str):
    with open(ptx_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if not line_to_insert.endswith('\n'):
        line_to_insert += '\n'

    lines.insert(ptx_line_num - 1, line_to_insert)
    with open(ptx_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def insert_multiple_line(ptx_file: str, ptx_line_num: int, line_list: List[str]):
    offset = 0
    for line in line_list:
        insert_a_line(ptx_file, ptx_line_num + offset, line)
        offset += 1
    return offset

def replace_mark(lines: List[str], mode, patch: str) -> List[str]:
    result = []
    for line in lines:
        new_line = re.sub(mode, patch, line)
        result.append(new_line)

    return result

def main():
    parser = argparse.ArgumentParser( 
        description="Insert probe to PTX"
    )
    parser.add_argument("ptx_file", help="Path to PTX file to be probed")
    parser.add_argument("--config", help="Path to config file")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    target_kernel = config['ominiscope']['target_kernel']
    target_file = int(config['ominiscope']['target_file'])
    buffer_param1 = int(config['ominiscope']['buffer_param1'])
    buffer_param2 = int(config['ominiscope']['buffer_param2'])
    dump = config['output']['dump_ptx']
    map = config['output']['loc_map']
    mode = config['ominiscope']['mode']
    t_start = int(config['ominiscope']['t_start'])
    t_end = int(config['ominiscope']['t_end'])

    head = config['probe']['head_ptx']
    entry = config['probe']['entry_ptx']
    exit = config['probe']['exit_ptx']

    with open(args.ptx_file, 'r', encoding='utf-8') as file:
        ptx_lines = file.readlines()
    
    if not is_right_kernel(ptx_lines, target_kernel):
        return
    
    with open(head, 'r', encoding='utf-8') as f:
        head_lines = f.readlines()
        total = t_end - t_start + 1
        head_lines = replace_mark(head_lines, r'KERNEL_NAME', f"{target_kernel}")
        head_lines = replace_mark(head_lines, r'PARAM1', f"{buffer_param1}")
        head_lines = replace_mark(head_lines, r'PARAM2', f"{buffer_param2}")
        head_lines = replace_mark(head_lines, r'START', f"{t_start}")
        head_lines = replace_mark(head_lines, r'END', f"{t_end}")
        head_lines = replace_mark(head_lines, r'TOTAL', f"{total}")
    
    with open(entry, 'r', encoding='utf-8') as f:
        entry_lines = f.readlines()

    with open(exit, 'r', encoding='utf-8') as f:
        exit_lines = f.readlines()
    
    loc_probe_map: dict[int, int] = {}
    sub_block_list = builder(ptx_lines)

    probe_pair_list = []
    i = 0
    if mode == "block":
        probe_pair = [None, None]
        for j, block in enumerate(sub_block_list):
            if block.is_start or block.is_end or block.is_sync or block.file != target_file:
                continue
            if i not in loc_probe_map:
                loc_probe_map[i] = []
            loc_probe_map[i] = block.loc
            if sub_block_list[j + 1].is_sync:
                probe_pair[0] = block.entry
            elif sub_block_list[j - 1].is_sync:
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
        probe_pair = [None, None]
        probe_pair[0] = sub_block_list[1].entry
        probe_pair[1] = sub_block_list[-2].exit
        probe_pair_list.append(probe_pair)
        loc_probe_map[i] = -1
    else:
        print(f"Unsupported mode:{mode}")
        sys.exit(1)

    dead_elimination = config.getboolean('dead_elimination', 'enable')
    if dead_elimination:
        active_list = config['dead_elimination']['active_list']
        
        with open(active_list, 'r', encoding='utf-8') as f:
            data = json.load(f)
            active_start = data['active_start']['list']
            active_end = data['active_end']['list']

            for i, pair in enumerate(probe_pair_list):
                new_pair = [None, None]
                if i in active_start:
                    new_pair[0] = pair[0]
                if i in active_end:
                    new_pair[1] = pair[1]
                probe_pair_list[i] = new_pair

    start_line = sub_block_list[0].entry
    offset = 0
    offset += insert_multiple_line(args.ptx_file, start_line, head_lines)

    for i, probe_pair in enumerate(probe_pair_list):
        if probe_pair[0] is not None:
            entry_insert = replace_mark(entry_lines, r'INDEX', f"{i}")
            offset += insert_multiple_line(args.ptx_file, probe_pair[0] + offset, entry_insert)
        if probe_pair[1] is not None:
            exit_insert = replace_mark(exit_lines, r'INDEX', f"{i}")
            offset += insert_multiple_line(args.ptx_file, probe_pair[1] + offset + 1, exit_insert)

    if map is not None:
        with open(map, 'w', encoding='utf-8') as f:
            json.dump(loc_probe_map, f, indent=2)

    if dump is not None:
        shutil.copyfile(args.ptx_file, dump)

if __name__ == '__main__':
    main()