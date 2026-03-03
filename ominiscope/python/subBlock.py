import re
from typing import List

START   = re.compile(r'^{$')
END     = re.compile(r'^\s*ret;')
LABEL   = re.compile(r'^\s*\$L__BB\d+_(\d+):')
BRA_CON = re.compile(r'^\s*@.*\$L__BB\d+_(\d+);\s*$')
BRA_UNI = re.compile(r'^\s*bra\.uni.*\$L__BB\d+_(\d+);\s*$')
LOC     = re.compile(r'^\s*\.loc\s+(\d+)\s+(\d+)\s+(\d+)')
SYNC    = re.compile(r'^\s*mbarrier\.try_wait\.parity\.acquire')


class subBlock:
    def __init__(self, entry, file, loc):
        self.entry     = entry
        self.exit      = 0
        self.file: int = file
        self.loc: int  = loc
        self.is_start  = False
        self.is_end    = False
        self.is_sync   = False

    def print_block(self):
        print(f"\nA sub block:")
        print(f"entry at line {self.entry}")
        print(f"exit at line {self.exit}")
        print(f"Source file is {self.file}, loc is {self.loc}")
    
    def __str__(self):
        return f"BB[{self.entry}-{self.exit}]"
    
    def __repr__(self):
        return self.__str__()


def builder(lines: List[str]) -> List[subBlock]:
    subBlock_list = []
    current_subBlock = None
    current_loc = -1
    current_file = 1

    i = 0
    while i < len(lines):
        ptx_line_num = i + 1
        line = lines[i].rstrip()

        start_match   = re.match(START, line)
        end_match     = re.match(END, line)
        label_match   = re.match(LABEL, line)
        bra_uni_match = re.match(BRA_UNI, line)
        bra_con_match = re.match(BRA_CON, line)
        loc_match     = re.match(LOC, line)
        sync_match    = re.match(SYNC, line)

        if start_match:
            current_subBlock = subBlock(ptx_line_num + 1, current_file, current_loc)
            current_subBlock.is_start = True
        elif loc_match:
            if current_subBlock.exit != ptx_line_num - 1:
                current_subBlock.exit = ptx_line_num - 1
                subBlock_list.append(current_subBlock)
            current_file = int(loc_match.group(1))
            current_loc = int(loc_match.group(2))
            current_subBlock = subBlock(ptx_line_num + 1, current_file, current_loc)
        elif sync_match:
            current_subBlock.is_sync = True
        elif label_match or bra_con_match or bra_uni_match:
            if current_subBlock.entry != ptx_line_num:
                current_subBlock.exit = ptx_line_num - 1
                subBlock_list.append(current_subBlock)
            current_subBlock = subBlock(ptx_line_num, current_file, current_loc)
            current_subBlock.exit = ptx_line_num
            subBlock_list.append(current_subBlock)
            if i + 1 < len(lines) and not re.match(LOC, lines[i + 1].rstrip()):
                current_subBlock = subBlock(ptx_line_num + 1, current_file, current_loc)
        elif end_match:
            current_subBlock.exit = ptx_line_num
            current_subBlock.is_end = True
            subBlock_list.append(current_subBlock)
            return subBlock_list
        
        i += 1
    return subBlock_list
