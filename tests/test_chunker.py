import pytest

from rag_chunker import chunk_markdown, estimate_tokens, split_sentences


def test_empty_document_yields_no_chunks():
    assert chunk_markdown("", max_tokens=512, overlap=0) == []


def test_single_paragraph_no_heading_produces_one_chunk():
    text = "Retry the job. Then check the logs."
    chunks = chunk_markdown(text, max_tokens=512, overlap=0)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.heading_path == ()
    assert chunk.body == text
    assert chunk.text == text
    assert chunk.start_line == 1
    assert chunk.end_line == 1
    assert chunk.token_estimate == estimate_tokens(text)
    assert chunk.oversized is False


def test_chunks_never_span_a_heading():
    text = "# A\n\nHi.\n\n# B\n\nBye.\n"
    chunks = chunk_markdown(text, max_tokens=512, overlap=0)
    assert len(chunks) == 2
    assert chunks[0].heading_path == ("A",)
    assert chunks[0].text == "A\n\nHi."
    assert chunks[1].heading_path == ("B",)
    assert chunks[1].text == "B\n\nBye."


def test_code_block_is_atomic_and_flagged_oversized_over_budget():
    code = (
        "```python\n"
        "print('this line is definitely long enough to exceed a tiny token budget')\n"
        "```"
    )
    chunks = chunk_markdown(code + "\n", max_tokens=5, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].body == code
    assert chunks[0].oversized is True
    assert (chunks[0].start_line, chunks[0].end_line) == (1, 3)


def test_table_is_kept_whole():
    table = "| A | B |\n| --- | --- |\n| 1 | 2 |"
    chunks = chunk_markdown(table + "\n", max_tokens=512, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].body == table
    assert chunks[0].oversized is False


def test_long_paragraph_splits_on_sentence_boundaries():
    # The first sentence is deliberately the longest, so it alone sets
    # the budget: no other single sentence can ever be oversized, and
    # the packer is guaranteed to close the first chunk right after it.
    text = (
        "Then escalate to the on-call engineer immediately. "
        "Check logs. Retry once. Close it."
    )
    sentences = split_sentences(text)
    assert len(sentences) == 4
    budget = estimate_tokens(sentences[0])

    chunks = chunk_markdown(text, max_tokens=budget, overlap=0)

    assert len(chunks) > 1
    assert chunks[0].body == sentences[0]
    assert not any(chunk.oversized for chunk in chunks)
    assert " ".join(chunk.body for chunk in chunks) == text


def test_overlap_repeats_trailing_sentence_in_next_chunk():
    text = "Aaa bbb ccc. Ddd eee fff. Ggg hhh iii. Jjj kkk lll."
    chunks = chunk_markdown(text, max_tokens=8, overlap=4)
    assert [chunk.body for chunk in chunks] == [
        "Aaa bbb ccc. Ddd eee fff.",
        "Ddd eee fff. Ggg hhh iii.",
        "Ggg hhh iii. Jjj kkk lll.",
    ]
    assert all(chunk.token_estimate == 8 for chunk in chunks)
    assert not any(chunk.oversized for chunk in chunks)


def test_overlap_resets_at_heading():
    text = (
        "# A\n\nAaa bbb ccc. Ddd eee fff. Ggg hhh iii.\n\n"
        "# B\n\nMmm nnn ooo. Ppp qqq rrr. Sss ttt uuu.\n"
    )
    chunks = chunk_markdown(text, max_tokens=11, overlap=4)
    assert [chunk.index for chunk in chunks] == [0, 1, 2, 3]
    assert [chunk.heading_path for chunk in chunks] == [
        ("A",),
        ("A",),
        ("B",),
        ("B",),
    ]
    assert chunks[0].body == "Aaa bbb ccc. Ddd eee fff."
    assert chunks[1].body == "Ddd eee fff. Ggg hhh iii."
    # Section B opens on its own first sentence, not the tail of A.
    assert chunks[2].body == "Mmm nnn ooo. Ppp qqq rrr."
    assert chunks[3].body == "Ppp qqq rrr. Sss ttt uuu."
    assert chunks[0].text == "A\n\nAaa bbb ccc. Ddd eee fff."
    assert chunks[2].text == "B\n\nMmm nnn ooo. Ppp qqq rrr."


def test_single_item_list_is_atomic_and_can_be_oversized():
    text = "- Only item here that is long enough to blow a tiny budget for sure\n"
    chunks = chunk_markdown(text, max_tokens=3, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].body == text.strip("\n")
    assert chunks[0].oversized is True


def test_list_items_split_across_chunks_when_the_list_is_long():
    text = "- Aaa bbb ccc\n- Ddd eee fff\n- Ggg hhh iii\n"
    chunks = chunk_markdown(text, max_tokens=4, overlap=0)
    assert [chunk.body for chunk in chunks] == [
        "- Aaa bbb ccc",
        "- Ddd eee fff",
        "- Ggg hhh iii",
    ]
    assert not any(chunk.oversized for chunk in chunks)


def test_to_dict_contains_expected_fields():
    chunks = chunk_markdown("Hello there.", max_tokens=512, overlap=0)
    assert chunks[0].to_dict() == {
        "index": 0,
        "text": "Hello there.",
        "heading_path": [],
        "start_line": 1,
        "end_line": 1,
        "token_estimate": chunks[0].token_estimate,
        "oversized": False,
    }


def test_max_tokens_must_be_positive():
    with pytest.raises(ValueError):
        chunk_markdown("text", max_tokens=0, overlap=0)


def test_overlap_must_not_be_negative():
    with pytest.raises(ValueError):
        chunk_markdown("text", max_tokens=10, overlap=-1)


def test_overlap_must_be_smaller_than_max_tokens():
    with pytest.raises(ValueError):
        chunk_markdown("text", max_tokens=10, overlap=10)
