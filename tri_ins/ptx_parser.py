import re
from typing import List, Optional

START   = re.compile(r'^{$')
END     = re.compile(r'^\s*ret;')
LABEL   = re.compile(r'^\s*\$L__BB\d+_(\d+):')
BRA_CON = re.compile(r'^\s*@.*\$L__BB\d+_(\d+);\s*$')
BRA_UNI = re.compile(r'^\s*bra\.uni.*\$L__BB\d+_(\d+);\s*$')
LOC     = re.compile(r'^\s*\.loc\s+(\d+)\s+(\d+)\s+(\d+)')
COMMENT = re.compile(r'^\s*//')
MMA     = re.compile(r'^\s*mma')
EMPTY   = re.compile(r'^\s*$')
SYNC    = re.compile(r'^\s*mbarrier\.try_wait\.parity\.acquire')


class sub_block:
    def __init__(self, entry: int, file: int, loc: int):
        self.entry: int = entry
        self.exit:  int = 0
        self.file:  int = file
        self.loc:   int = loc
        self.mma_lines:   List[int] = []
        self.empty_lines: List[int] = []
        self.is_start: bool = False
        self.is_end:   bool = False
        self.is_sync:  bool = False

    def print_block(self):
        print(f"\nA sub block:")
        print(f"  entry at line {self.entry}")
        print(f"  exit  at line {self.exit}")
        flags = []
        if self.is_start: flags.append("START")
        if self.is_end:   flags.append("END")
        if self.is_sync:  flags.append("SYNC")
        print(f"  empty={self.empty_lines}, mma={self.mma_lines}, "
              f"file={self.file}, loc={self.loc}"
              + (f"  [{','.join(flags)}]" if flags else ""))

    def __str__(self):
        return f"BB[{self.entry}-{self.exit}]"

    def __repr__(self):
        return self.__str__()


def builder(lines: List[str]) -> List['sub_block']:
    sub_block_list: List[sub_block] = []
    current_sub_block: Optional[sub_block] = None
    current_loc:  int = -1
    current_file: int = 1

    i = 0
    while i < len(lines):
        ptx_line_num = i + 1
        line = lines[i].rstrip()

        start_match   = re.match(START,   line)
        end_match     = re.match(END,     line)
        label_match   = re.match(LABEL,   line)
        bra_uni_match = re.match(BRA_UNI, line)
        bra_con_match = re.match(BRA_CON, line)
        loc_match     = re.match(LOC,     line)
        comment_match = re.match(COMMENT, line)
        mma_match     = re.match(MMA,     line)
        empty_match   = re.match(EMPTY,   line)
        sync_match    = re.match(SYNC,    line)

        if start_match:
            current_sub_block = sub_block(ptx_line_num + 1, current_file, current_loc)
            current_sub_block.is_start = True

        elif current_sub_block is None:
            pass

        elif loc_match:
            if current_sub_block.exit != ptx_line_num - 1:
                current_sub_block.exit = ptx_line_num - 1
                sub_block_list.append(current_sub_block)
            current_file = int(loc_match.group(1))
            current_loc  = int(loc_match.group(2))
            current_sub_block = sub_block(ptx_line_num + 1, current_file, current_loc)

        elif comment_match or empty_match:
            current_sub_block.empty_lines.append(ptx_line_num)

        elif mma_match:
            current_sub_block.mma_lines.append(ptx_line_num)

        elif sync_match:
            current_sub_block.is_sync = True

        elif label_match or bra_con_match or bra_uni_match:
            if current_sub_block.entry != ptx_line_num:
                current_sub_block.exit = ptx_line_num - 1
                sub_block_list.append(current_sub_block)
            boundary_block = sub_block(ptx_line_num, current_file, current_loc)
            boundary_block.exit = ptx_line_num
            sub_block_list.append(boundary_block)
            if i + 1 < len(lines) and not re.match(LOC, lines[i + 1].rstrip()):
                current_sub_block = sub_block(ptx_line_num + 1, current_file, current_loc)
            else:
                current_sub_block = boundary_block

        elif end_match:
            current_sub_block.exit = ptx_line_num
            current_sub_block.is_end = True
            sub_block_list.append(current_sub_block)
            return sub_block_list

        i += 1
