import json

from rag_chunker import chunk_markdown, chunks_to_jsonl
from rag_chunker.serialize import chunks_to_json_array


def test_chunks_to_jsonl_empty_list_is_empty_string():
    assert chunks_to_jsonl([]) == ""


def test_chunks_to_jsonl_one_line_per_chunk():
    text = "# A\n\nHi.\n\n# B\n\nBye.\n"
    chunks = chunk_markdown(text, max_tokens=512, overlap=0)

    rendered = chunks_to_jsonl(chunks)
    lines = rendered.split("\n")

    assert len(lines) == 2
    assert [json.loads(line) for line in lines] == [chunk.to_dict() for chunk in chunks]


def test_chunks_to_json_array_round_trips_to_dict():
    chunks = chunk_markdown("Hello there.", max_tokens=512, overlap=0)

    parsed = json.loads(chunks_to_json_array(chunks))

    assert parsed == [chunk.to_dict() for chunk in chunks]


def test_chunks_to_json_array_empty_list_is_empty_array():
    assert chunks_to_json_array([]) == "[]"
