"""
PTX Sub-Block Parser (adapted for Triton)
==========================================
Based on Exist_Package/sub_block.py, adapted to handle Triton-generated PTX.

Key differences from cuTile PTX:
  - Triton uses bar.sync (not mbarrier.try_wait.parity.acquire) for synchronization
  - Triton has additional labels ($L__tmp*, $L__func_*) for debug info
  - Label format $L__BB0_N is the same
"""

import re
from typing import List, Optional

# ---- Regex patterns for PTX parsing ----
START = re.compile(r'^{$')
END = re.compile(r'^\s*ret;')
LABEL = re.compile(r'\$L__BB\d+_(\d+)')
BRA_CON = re.compile(r'^\s*@.*\$L__BB\d+_(\d+);\s*$')
BRA_UNI = re.compile(r'^\s*bra\.uni.*\$L__BB\d+_(\d+);\s*$')
LOC = re.compile(r'^\s*\.loc\s+(\d+)\s+(\d+)\s+(\d+)')
COMMENT = re.compile(r'^\s*//')
MMA = re.compile(r'^\s*mma')
EMPTY = re.compile(r'^\s*$')
# Triton uses bar.sync; cuTile used mbarrier.try_wait.parity.acquire
SYNC = re.compile(r'^\s*(mbarrier\.try_wait\.parity\.acquire|bar\.sync)')
# Debug info labels that should be ignored (not real basic block boundaries)
DEBUG_LABEL = re.compile(r'^\$L__(func_|tmp)')


class sub_block:
    def __init__(self, entry: int, loc: int):
        self.entry: int = entry        # 1-based line number
        self.exit: int = 0             # 1-based line number
        self.loc: int = loc            # Python source line number from .loc
        self.mma_lines: List[int] = []
        self.empty_lines: List[int] = []
        self.is_start: bool = False
        self.is_end: bool = False
        self.is_sync: bool = False
        self.is_bra: bool = False      # this 1-line block is a branch instruction
        self.is_label: bool = False    # this 1-line block is a label

    def print_block(self):
        flags = []
        if self.is_start:
            flags.append("START")
        if self.is_end:
            flags.append("END")
        if self.is_sync:
            flags.append("SYNC")
        if self.is_bra:
            flags.append("BRA")
        if self.is_label:
            flags.append("LABEL")
        flag_str = f" [{','.join(flags)}]" if flags else ""
        print(f"  BB[{self.entry}-{self.exit}] loc={self.loc} "
              f"mma={len(self.mma_lines)} empty={len(self.empty_lines)}{flag_str}")

    def __str__(self):
        return f"BB[{self.entry}-{self.exit}]"

    def __repr__(self):
        return self.__str__()


def builder(lines: List[str]) -> List[sub_block]:
    """
    Parse PTX lines into sub-blocks based on .loc directives, labels, and branches.

    Args:
        lines: List of PTX lines (each ending with newline)

    Returns:
        List of sub_block objects representing the control flow structure
    """
    sub_block_list: List[sub_block] = []
    current_sub_block: Optional[sub_block] = None
    current_loc: int = -1

    i = 0
    while i < len(lines):
        ptx_line_num = i + 1
        line = lines[i].rstrip()

        start_match = re.match(START, line)
        end_match = re.match(END, line)
        label_match = re.search(LABEL, line)
        bra_uni_match = re.match(BRA_UNI, line)
        bra_con_match = re.match(BRA_CON, line)
        loc_match = re.match(LOC, line)
        comment_match = re.match(COMMENT, line)
        mma_match = re.match(MMA, line)
        empty_match = re.match(EMPTY, line)
        sync_match = re.match(SYNC, line)
        debug_label_match = re.match(DEBUG_LABEL, line)

        # Skip debug-info labels ($L__tmp*, $L__func_*) — they are not real BB boundaries
        if debug_label_match:
            if current_sub_block is not None:
                current_sub_block.empty_lines.append(ptx_line_num)
            i += 1
            continue

        if start_match:
            current_sub_block = sub_block(ptx_line_num + 1, current_loc)
            current_sub_block.is_start = True
        elif current_sub_block is None:
            # Haven't entered the kernel body yet
            i += 1
            continue
        elif loc_match:
            if current_sub_block.exit != ptx_line_num - 1:
                current_sub_block.exit = ptx_line_num - 1
                sub_block_list.append(current_sub_block)
            current_loc = int(loc_match.group(2))
            current_sub_block = sub_block(ptx_line_num + 1, current_loc)
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
            boundary_block = sub_block(ptx_line_num, current_loc)
            boundary_block.exit = ptx_line_num
            if bra_con_match or bra_uni_match:
                boundary_block.is_bra = True
            else:
                boundary_block.is_label = True
            sub_block_list.append(boundary_block)
            if i + 1 < len(lines) and not re.match(LOC, lines[i + 1]):
                current_sub_block = sub_block(ptx_line_num + 1, current_loc)
            else:
                current_sub_block = boundary_block  # will be replaced by loc handler
        elif end_match:
            current_sub_block.exit = ptx_line_num
            current_sub_block.is_end = True
            sub_block_list.append(current_sub_block)
            return sub_block_list

        i += 1

    # If we get here without seeing ret, still return what we have
    if current_sub_block is not None and current_sub_block not in sub_block_list:
        current_sub_block.exit = len(lines)
        current_sub_block.is_end = True
        sub_block_list.append(current_sub_block)

    return sub_block_list
