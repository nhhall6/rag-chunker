"""Command-line entry point: ``rag-chunker`` / ``python -m rag_chunker.cli``.

Thin argument handling over :func:`chunk_markdown`; the chunking logic
itself lives in :mod:`chunker`.
"""

import argparse
import json
import sys

from .chunker import chunk_markdown
from .serialize import chunks_to_json_array, chunks_to_jsonl
from .tokens import estimate_tokens

# Keys a --config file may set, and the type each one must have. Matches
# the argparse dest names 1:1 so the values can be handed straight to
# parser.set_defaults(). "input" is deliberately excluded: it names the
# document, which differs on every invocation.
_CONFIG_SPEC = {
    "max_tokens": int,
    "overlap": int,
    "no_heading_prefix": bool,
    "array": bool,
    "stats": bool,
    "output": str,
}


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
    parser.add_argument(
        "--config", metavar="PATH",
        help="JSON file of default option values; any flag given on the "
             "command line still overrides it",
    )
    return parser


def _load_config(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: {error}") from error

    if not isinstance(data, dict):
        raise ValueError(f"{path}: config file must contain a JSON object")

    for key, value in data.items():
        if key not in _CONFIG_SPEC:
            raise ValueError(f"{path}: unknown config key {key!r}")
        expected = _CONFIG_SPEC[key]
        # bool is a subclass of int, so max_tokens/overlap would silently
        # accept true/false without this extra check.
        valid = isinstance(value, expected) and not (expected is int and isinstance(value, bool))
        if not valid:
            raise ValueError(f"{path}: {key!r} must be a {expected.__name__}")

    return data


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
    if argv is None:
        argv = sys.argv[1:]

    parser = build_parser()

    # --config has to be known before the real parse so its values can
    # seed the defaults that real flags on the command line then override.
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--config")
    config_path = peek.parse_known_args(argv)[0].config

    if config_path:
        try:
            parser.set_defaults(**_load_config(config_path))
        except (OSError, ValueError) as error:
            print(f"rag-chunker: {error}", file=sys.stderr)
            return 1

    args = parser.parse_args(argv)

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
