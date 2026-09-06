"""Command-line entry point: ``rag-chunker`` / ``python -m rag_chunker.cli``.

Thin argument handling over :func:`chunk_markdown`; the chunking logic
itself lives in :mod:`chunker`.
"""

import argparse
import sys

from .chunker import chunk_markdown
from .serialize import chunks_to_json_array, chunks_to_jsonl
from .tokens import estimate_tokens


def build_parser():
    parser = argparse.ArgumentParser(
        prog="rag-chunker",
        description="Split a markdown document into embedding-sized chunks.",
    )
    parser.add_argument("input", help="path to a markdown file, or - for stdin")
    parser.add_argument(
        "--max-tokens", type=int, default=512, metavar="N",
        help="chunk size ceiling, heading prefix included (default: 512)",
    )
    parser.add_argument(
        "--overlap", type=int, default=64, metavar="N",
        help="trailing tokens repeated in the next chunk of a section (default: 64)",
    )
    parser.add_argument(
        "--no-heading-prefix", action="store_true",
        help="do not prepend the heading path to the chunk text",
    )
    parser.add_argument(
        "--array", action="store_true",
        help="emit one indented JSON array instead of JSON lines",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="print a size summary to stderr",
    )
    parser.add_argument(
        "-o", dest="output", metavar="PATH",
        help="write the result to a file instead of stdout",
    )
    return parser


def _read_source(path):
    if path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _strip_heading_prefix(chunks):
    for chunk in chunks:
        chunk.text = chunk.body
        chunk.token_estimate = estimate_tokens(chunk.body)


def _stats_line(chunks):
    if not chunks:
        return "0 chunks"
    tokens = [chunk.token_estimate for chunk in chunks]
    oversized = sum(1 for chunk in chunks if chunk.oversized)
    average = sum(tokens) // len(tokens)
    return (
        f"{len(chunks)} chunks | tokens min {min(tokens)} avg {average} "
        f"max {max(tokens)} | {oversized} oversized"
    )


def main(argv=None):
    args = build_parser().parse_args(argv)

    try:
        text = _read_source(args.input)
    except OSError as error:
        print(f"rag-chunker: {error}", file=sys.stderr)
        return 1

    try:
        chunks = chunk_markdown(text, max_tokens=args.max_tokens, overlap=args.overlap)
    except ValueError as error:
        print(f"rag-chunker: {error}", file=sys.stderr)
        return 1

    if args.no_heading_prefix:
        _strip_heading_prefix(chunks)

    rendered = chunks_to_json_array(chunks) if args.array else chunks_to_jsonl(chunks)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    else:
        print(rendered)

    if args.stats:
        print(_stats_line(chunks), file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
