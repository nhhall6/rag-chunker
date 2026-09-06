"""Rendering chunks for output.

Kept separate from :mod:`chunker` because a chunk list has more than one
reasonable on-disk shape (newline-delimited vs. a single array), and
neither shape belongs on the ``Chunk`` dataclass itself.
"""

import json


def chunks_to_jsonl(chunks):
    """Render ``chunks`` as newline-delimited JSON, one object per line.

    Returns an empty string for an empty list rather than a single blank
    line, so callers can tell "no chunks" from "one empty chunk" would if
    that ever existed.
    """
    return "\n".join(json.dumps(chunk.to_dict(), ensure_ascii=False) for chunk in chunks)


def chunks_to_json_array(chunks):
    """Render ``chunks`` as one indented JSON array."""
    return json.dumps([chunk.to_dict() for chunk in chunks], ensure_ascii=False, indent=2)
