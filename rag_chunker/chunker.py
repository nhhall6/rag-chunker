"""Markdown-to-chunk conversion.

Turns parsed blocks into sized chunks for embedding: headings set the
section a chunk belongs to and are never split across, code and table
blocks are kept whole, and paragraphs fall back to sentence boundaries
when they do not fit. See the README for the shape this is trying to
produce.
"""

from dataclasses import dataclass
from typing import Tuple

from .blocks import _BULLET_RE, _ORDERED_RE, parse_blocks
from .sentences import split_sentences
from .tokens import estimate_tokens

_HEADING_SEP = " > "


@dataclass
class Chunk:
    """One packed chunk of a document, ready to embed.

    ``text`` is the heading prefix (if any) followed by ``body``; embed
    ``text``, not ``body``, so retrieval carries the section context.
    ``oversized`` is true only when a single piece of content that
    cannot be split further -- a code block, a table, a one-item list,
    or a lone sentence -- exceeds ``max_tokens`` on its own.
    """

    index: int
    text: str
    body: str
    heading_path: Tuple[str, ...]
    start_line: int
    end_line: int
    token_estimate: int
    oversized: bool

    def to_dict(self):
        return {
            "index": self.index,
            "text": self.text,
            "heading_path": list(self.heading_path),
            "start_line": self.start_line,
            "end_line": self.end_line,
            "token_estimate": self.token_estimate,
            "oversized": self.oversized,
        }


@dataclass
class _Piece:
    """One unit a chunk is packed from: a sentence, a list item, or a
    whole atomic block. Pieces sharing ``group_id`` came from the same
    source block, so adjacent pieces in a group are rejoined with
    ``sep`` instead of the blank-line break used between blocks.
    """

    text: str
    start_line: int
    end_line: int
    group_id: object
    sep: str
    atomic: bool


def chunk_markdown(text, max_tokens=512, overlap=64):
    """Split ``text`` into a list of :class:`Chunk`.

    A chunk never spans a heading boundary. Within a section, blocks
    are packed greedily up to ``max_tokens``: a code block or table is
    always emitted whole (and flagged ``oversized`` if that alone
    exceeds the budget), and an overlong paragraph is packed sentence
    by sentence instead, an overlong list item by item. ``overlap``
    trailing tokens of prose are repeated at the start of the next
    chunk of the same section, and the carry resets at every heading.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens")

    chunks = []
    for heading_path, section_blocks in _group_into_sections(parse_blocks(text)):
        chunks.extend(_pack_section(section_blocks, heading_path, max_tokens, overlap))

    for position, chunk in enumerate(chunks):
        chunk.index = position

    return chunks


def _group_into_sections(blocks):
    """Group blocks into ``(heading_path, blocks)`` runs.

    ``heading_path`` holds the titles of the enclosing headings,
    innermost last, tracked with a stack keyed on heading level so a
    heading closes any open heading at its level or deeper but leaves
    its shallower ancestors in place.
    """
    sections = []
    stack = []
    current_blocks = []
    current_path = ()

    for block in blocks:
        if block.kind == "heading":
            if current_blocks:
                sections.append((current_path, current_blocks))
            current_blocks = []
            while stack and stack[-1][0] >= block.level:
                stack.pop()
            stack.append((block.level, block.text.lstrip("#").strip()))
            current_path = tuple(title for _, title in stack)
        else:
            current_blocks.append(block)

    if current_blocks:
        sections.append((current_path, current_blocks))

    return sections


def _pack_section(blocks, heading_path, max_tokens, overlap):
    pieces = _flatten_blocks(blocks)
    if not pieces:
        return []

    heading_prefix = _HEADING_SEP.join(heading_path)
    chunks = []
    overlap_pieces = []
    new_pieces = []

    for piece in pieces:
        candidate = overlap_pieces + new_pieces + [piece]
        if new_pieces and _rendered_tokens(heading_prefix, candidate) <= max_tokens:
            new_pieces.append(piece)
            continue

        if new_pieces:
            chunks.append(
                _finalize_chunk(heading_path, heading_prefix, overlap_pieces, new_pieces, max_tokens)
            )
            overlap_pieces = _trailing_overlap_pieces(new_pieces, overlap)

        if overlap_pieces and _rendered_tokens(heading_prefix, overlap_pieces + [piece]) > max_tokens:
            overlap_pieces = []

        new_pieces = [piece]

    chunks.append(_finalize_chunk(heading_path, heading_prefix, overlap_pieces, new_pieces, max_tokens))
    return chunks


def _flatten_blocks(blocks):
    pieces = []
    for block in blocks:
        if block.kind in ("code", "table"):
            pieces.append(_Piece(block.text, block.start_line, block.end_line, object(), "", True))
            continue

        if block.kind == "list":
            items = _split_list_items(block.text)
            if len(items) <= 1:
                pieces.append(_Piece(block.text, block.start_line, block.end_line, object(), "", True))
            else:
                group_id = object()
                for item in items:
                    pieces.append(_Piece(item, block.start_line, block.end_line, group_id, "\n", False))
            continue

        group_id = object()
        for sentence in split_sentences(block.text):
            pieces.append(_Piece(sentence, block.start_line, block.end_line, group_id, " ", False))

    return pieces


def _split_list_items(text):
    """Split a list block into its top-level items.

    A line starting a bullet or ordered marker begins a new item;
    everything up to the next marker, including indented continuation
    lines, stays attached to that item.
    """
    items = []
    current = []
    for line in text.split("\n"):
        if _BULLET_RE.match(line) or _ORDERED_RE.match(line):
            if current:
                items.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        items.append("\n".join(current))
    return items


def _render_body(pieces):
    if not pieces:
        return ""
    parts = [pieces[0].text]
    for previous, piece in zip(pieces, pieces[1:]):
        parts.append(piece.sep if piece.group_id == previous.group_id else "\n\n")
        parts.append(piece.text)
    return "".join(parts)


def _rendered_tokens(heading_prefix, pieces):
    body = _render_body(pieces)
    text = f"{heading_prefix}\n\n{body}" if heading_prefix else body
    return estimate_tokens(text)


def _trailing_overlap_pieces(new_pieces, overlap):
    """Pick the trailing non-atomic pieces of a finished chunk to carry
    into the next one, stopping once ``overlap`` tokens are covered but
    always keeping at least one piece so overlap is never silently
    dropped for a short final sentence.
    """
    if overlap <= 0:
        return []

    eligible = [piece for piece in new_pieces if not piece.atomic]
    if not eligible:
        return []

    collected = []
    total = 0
    for piece in reversed(eligible):
        piece_tokens = estimate_tokens(piece.text)
        if collected and total + piece_tokens > overlap:
            break
        collected.append(piece)
        total += piece_tokens
        if total >= overlap:
            break

    collected.reverse()
    return collected


def _finalize_chunk(heading_path, heading_prefix, overlap_pieces, new_pieces, max_tokens):
    body = _render_body(overlap_pieces + new_pieces)
    text = f"{heading_prefix}\n\n{body}" if heading_prefix else body
    own_tokens = _rendered_tokens(heading_prefix, new_pieces)

    return Chunk(
        index=0,
        text=text,
        body=body,
        heading_path=heading_path,
        start_line=min(piece.start_line for piece in new_pieces),
        end_line=max(piece.end_line for piece in new_pieces),
        token_estimate=estimate_tokens(text),
        oversized=own_tokens > max_tokens,
    )
