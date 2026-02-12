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
    buffer_param = int(config['ominiscope']['buffer_param'])
    dump = config['ominiscope']['dump_ptx']
    map = config['ominiscope']['loc_map']
    mode = config['ominiscope']['mode']
    t_start = int(config['ominiscope']['t_start'])
    t_end = int(config['ominiscope']['t_end'])

    head = config['ominiscope']['head_ptx']
    entry = config['ominiscope']['entry_ptx']
    exit = config['ominiscope']['exit_ptx']
    config = config['ominiscope']['config_ptx']

    with open(args.ptx_file, 'r', encoding='utf-8') as file:
        ptx_lines = file.readlines()
    
    if not is_right_kernel(ptx_lines, target_kernel):
        return
    
    with open(head, 'r', encoding='utf-8') as f:
        head_lines = f.readlines()
        total = t_end - t_start + 1
        head_lines = replace_mark(head_lines, r'KERNEL_NAME', f"{target_kernel}")
        head_lines = replace_mark(head_lines, r'PARAM', f"{buffer_param}")
        head_lines = replace_mark(head_lines, r'START', f"{t_start}")
        head_lines = replace_mark(head_lines, r'END', f"{t_end}")
        head_lines = replace_mark(head_lines, r'TOTAL', f"{total}")
    
    with open(entry, 'r', encoding='utf-8') as f:
        entry_lines = f.readlines()

    with open(exit, 'r', encoding='utf-8') as f:
        exit_lines = f.readlines()
    
    with open(config, 'r', encoding='utf-8') as f:
        config_lines = f.readlines()
        config_lines = replace_mark(config_lines, r'KERNEL_NAME', f"{target_kernel}")
        config_lines = replace_mark(config_lines, r'PARAM', f"{buffer_param}")

    loc_probe_map: dict[int, int] = {}
    sub_block_list = builder(ptx_lines)

    start_line = sub_block_list[0].entry
    offset = 0
    if mode == "block" or mode == "single" or mode == "entire":
        offset += insert_multiple_line(args.ptx_file, start_line, head_lines)
    elif mode == "config":
        offset += insert_multiple_line(args.ptx_file, start_line, config_lines)

    i = 1
    if mode == "block":
        for j, block in enumerate(sub_block_list):
            if block.is_start or block.is_end or block.is_sync:
                continue
            if i not in loc_probe_map:
                loc_probe_map[i] = []
            loc_probe_map[i] = block.loc
            if sub_block_list[j + 1].is_sync:
                entry_insert = replace_mark(entry_lines, r'INDEX', f"{i}")
                offset += insert_multiple_line(args.ptx_file, block.entry + offset, entry_insert)
            elif sub_block_list[j - 1].is_sync:
                exit_insert = replace_mark(exit_lines, r'INDEX', f"{i}")
                offset += insert_multiple_line(args.ptx_file, block.exit + offset + 1, exit_insert)
                i += 1
            else:
                entry_insert = replace_mark(entry_lines, r'INDEX', f"{i}")
                exit_insert = replace_mark(exit_lines, r'INDEX', f"{i}")
                offset += insert_multiple_line(args.ptx_file, block.entry + offset, entry_insert)
                offset += insert_multiple_line(args.ptx_file, block.exit + offset + 1, exit_insert)
                i += 1
    elif mode == "single":
        for block in sub_block_list:
            if i not in loc_probe_map:
                loc_probe_map[i] = []
            if block.is_start or block.is_end or block.is_sync:
                continue
            j = block.entry
            while j <= block.exit:
                if j in block.mma_lines:
                    if j - 5 not in block.mma_lines:
                        entry_insert = replace_mark(entry_lines, r'INDEX', f"{i}")
                        offset += insert_multiple_line(args.ptx_file, j + offset, entry_insert)
                        loc_probe_map[i] = block.loc
                        j += 5
                        continue
                    if j + 5 in block.mma_lines:
                        j += 5
                        continue
                    exit_insert = replace_mark(exit_lines, r'INDEX', f"{i}")
                    offset += insert_multiple_line(args.ptx_file, j + offset + 5, exit_insert)
                    i += 1
                    j += 5
                elif j in block.empty_lines:
                    j += 1
                else:
                    entry_insert = replace_mark(entry_lines, r'INDEX', f"{i}")
                    exit_insert = replace_mark(exit_lines, r'INDEX', f"{i}")
                    offset += insert_multiple_line(args.ptx_file, j + offset, entry_insert)
                    offset += insert_multiple_line(args.ptx_file, j + offset + 1, exit_insert)
                    loc_probe_map[i] = block.loc
                    i += 1
                    j += 1
    elif mode == "entire":
        entry_insert = replace_mark(entry_lines, r'INDEX', f"{i}")
        exit_insert = replace_mark(exit_lines, r'INDEX', f"{i}")
        offset += insert_multiple_line(args.ptx_file, sub_block_list[1].entry + offset, entry_insert)
        offset += insert_multiple_line(args.ptx_file, sub_block_list[-2].exit + offset + 1, exit_insert)
        loc_probe_map[i] = -1
    elif mode == "config":
        pass
    else:
        print(f"Unsupported mode:{mode}")
        sys.exit(1)

    if map is not None:
        with open(map, 'w', encoding='utf-8') as f:
            json.dump(loc_probe_map, f, indent=2)

    if dump is not None:
        shutil.copyfile(args.ptx_file, dump)

if __name__ == '__main__':
    main()