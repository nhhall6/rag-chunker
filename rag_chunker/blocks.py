"""Markdown structure parsing.

Splits a document into the shapes a chunker needs to respect: headings,
fenced code, pipe tables, list runs, and paragraphs. This is not a full
CommonMark parser, just enough structure recognition to keep a chunker
from cutting a block in the wrong place. Inline syntax (emphasis, links,
inline code) is left untouched inside block text.
"""

import re
from dataclasses import dataclass
from typing import Optional

_HEADING_RE = re.compile(r"^(#{1,6})(?:\s+(.*))?$")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_BULLET_RE = re.compile(r"^\s{0,3}[-*+]\s+")
_ORDERED_RE = re.compile(r"^\s{0,3}\d{1,9}[.)]\s+")

# A table separator row: cells made only of dashes and optional leading
# or trailing colons (for alignment), joined by pipes.
_TABLE_CELL_RE = re.compile(r"^:?-+:?$")


@dataclass
class Block:
    """One structural unit of a markdown document.

    ``kind`` is one of ``heading``, ``paragraph``, ``list``, ``code`` or
    ``table``. Line numbers are 1-based and inclusive, pointing back at
    the source text so a chunk can report where it came from.
    """

    kind: str
    text: str
    start_line: int
    end_line: int
    level: Optional[int] = None


def _is_table_separator(line):
    stripped = line.strip()
    if "|" not in stripped or not stripped:
        return False
    if not set(stripped) <= set("|:- "):
        return False
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    if not cells or any(not cell for cell in cells):
        return False
    return all(_TABLE_CELL_RE.match(cell) for cell in cells)


def _is_block_start(line, next_line):
    if _HEADING_RE.match(line) or _FENCE_RE.match(line):
        return True
    if _BULLET_RE.match(line) or _ORDERED_RE.match(line):
        return True
    if "|" in line and next_line is not None and _is_table_separator(next_line):
        return True
    return False


def parse_blocks(text):
    """Split ``text`` into a list of :class:`Block` in source order.

    Blank lines separate blocks but never appear inside one. A fenced
    code block that never finds its closing fence runs to end of file
    rather than swallowing the rest of the document as its content.
    """
    lines = text.splitlines()
    blocks = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        if not line.strip():
            i += 1
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            level = len(heading_match.group(1))
            blocks.append(Block("heading", line, i + 1, i + 1, level=level))
            i += 1
            continue

        fence_match = _FENCE_RE.match(line)
        if fence_match:
            fence_marker = fence_match.group(1)
            fence_char = fence_marker[0]
            fence_len = len(fence_marker)
            start = i
            i += 1
            while i < n:
                closing = _FENCE_RE.match(lines[i])
                if (
                    closing
                    and closing.group(1)[0] == fence_char
                    and len(closing.group(1)) >= fence_len
                    and not closing.group(2).strip()
                ):
                    i += 1
                    break
                i += 1
            blocks.append(
                Block("code", "\n".join(lines[start:i]), start + 1, i)
            )
            continue

        if "|" in line and i + 1 < n and _is_table_separator(lines[i + 1]):
            start = i
            i += 2
            while i < n and lines[i].strip() and "|" in lines[i]:
                i += 1
            blocks.append(
                Block("table", "\n".join(lines[start:i]), start + 1, i)
            )
            continue

        if _BULLET_RE.match(line) or _ORDERED_RE.match(line):
            start = i
            i += 1
            while i < n and lines[i].strip() and (
                _BULLET_RE.match(lines[i])
                or _ORDERED_RE.match(lines[i])
                or lines[i][:1].isspace()
            ):
                i += 1
            blocks.append(
                Block("list", "\n".join(lines[start:i]), start + 1, i)
            )
            continue

        start = i
        i += 1
        while i < n and lines[i].strip():
            next_line = lines[i + 1] if i + 1 < n else None
            if _is_block_start(lines[i], next_line):
                break
            i += 1
        blocks.append(
            Block("paragraph", "\n".join(lines[start:i]), start + 1, i)
        )

    return blocks
