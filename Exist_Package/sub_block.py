import re
from typing import List

START = re.compile(r'^{$')
END = re.compile(r'^\s*ret;')
LABEL = re.compile(r'\$L__BB0_(\d+)')
BRA_CON = re.compile(r'^\s*@.*\$L__BB0_(\d+);\s*$')
BRA_UNI = re.compile(r'^\s*bra\.uni.*\$L__BB0_(\d+);\s*$')
LOC = re.compile(r'^\s*\.loc\s+(\d+)\s+(\d+)\s+(\d+)')
COMMENT = re.compile(r'^\s*//')
MMA = re.compile(r'^\s*mma')
EMPTY = re.compile(r'^\s*$')
SYNC = re.compile(r'^\s*mbarrier.try_wait.parity.acquire')

class sub_block:
    def __init__(self, entry, loc):
        self.entry = entry
        self.exit = 0
        self.loc: int = loc
        self.mma_lines = []
        self.empty_lines = []
        self.is_start = False
        self.is_end = False
        self.is_sync = False

    def print_block(self):
        print(f"\nA sub block:")
        print(f"entry at line {self.entry}")
        print(f"exit at line {self.exit}")
        print(f"Having empty lines at {self.empty_lines}, mma lines at {self.mma_lines}, loc is {self.loc}")
    
    def __str__(self):
        return f"BB[{self.entry}-{self.exit}]"
    
    def __repr__(self):
        return self.__str__()

def builder(lines: List[str]) -> List[sub_block]:
    sub_block_list = []
    current_sub_block = None
    current_loc = -1

    i = 0
    while i <= len(lines):
        ptx_line_num = i + 1
        line = lines[i].rstrip()
        start_match = re.match(START, line)
        end_match = re.match(END, line)
        label_match = re.match(LABEL, line)
        bra_uni_match = re.match(BRA_UNI, line)
        bra_con_match = re.match(BRA_CON, line)
        loc_match = re.match(LOC, line)
        comment_match = re.match(COMMENT, line)
        mma_match = re.match(MMA, line)
        empty_match = re.match(EMPTY, line)
        sync_match = re.match(SYNC, line)

        if start_match:
            current_sub_block = sub_block(ptx_line_num + 1, current_loc)
            current_sub_block.is_start = True
        elif loc_match:
            if current_sub_block.exit != ptx_line_num - 1:
                current_sub_block.exit = ptx_line_num - 1
                sub_block_list.append(current_sub_block)
            current_loc = int(int(loc_match.group(2)))
            current_sub_block = sub_block(ptx_line_num + 1, current_loc)
        elif comment_match or empty_match:
            if current_sub_block is not None:
                current_sub_block.empty_lines.append(ptx_line_num)
        elif mma_match:
            current_sub_block.mma_lines.append(ptx_line_num)
        elif sync_match:
            current_sub_block.is_sync = True
        elif label_match or bra_con_match or bra_uni_match:
            if current_sub_block.entry != ptx_line_num:
                current_sub_block.exit = ptx_line_num - 1
                sub_block_list.append(current_sub_block)
            current_sub_block = sub_block(ptx_line_num, current_loc)
            current_sub_block.exit = ptx_line_num
            sub_block_list.append(current_sub_block)
            if not re.match(LOC, lines[i + 1]):
                current_sub_block = sub_block(ptx_line_num + 1, current_loc)
        elif end_match:
            current_sub_block.exit = ptx_line_num
            current_sub_block.is_end = True
            sub_block_list.append(current_sub_block)
            return sub_block_list
        
        i += 1

