from rag_chunker import parse_blocks


def _kinds(blocks):
    return [block.kind for block in blocks]


def test_empty_text_yields_no_blocks():
    assert parse_blocks("") == []


def test_blank_lines_separate_blocks_without_producing_their_own():
    text = "First paragraph.\n\n\nSecond paragraph.\n"
    blocks = parse_blocks(text)
    assert _kinds(blocks) == ["paragraph", "paragraph"]
    assert blocks[0].text == "First paragraph."
    assert blocks[1].text == "Second paragraph."


def test_atx_heading_levels_and_content():
    text = "# One\n## Two\n###### Six\n"
    blocks = parse_blocks(text)
    assert [(b.kind, b.level, b.text) for b in blocks] == [
        ("heading", 1, "# One"),
        ("heading", 2, "## Two"),
        ("heading", 6, "###### Six"),
    ]


def test_seven_hashes_is_not_a_heading():
    blocks = parse_blocks("####### Too many\n")
    assert blocks[0].kind == "paragraph"


def test_hash_without_space_is_not_a_heading():
    blocks = parse_blocks("#nohashspace\n")
    assert blocks[0].kind == "paragraph"


def test_heading_line_numbers_are_one_based():
    text = "\n\n# Third line\n"
    blocks = parse_blocks(text)
    assert blocks[0].start_line == 3
    assert blocks[0].end_line == 3


def test_fenced_code_block_is_captured_whole():
    text = "```python\nprint('hi')\n```\n"
    blocks = parse_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].kind == "code"
    assert blocks[0].text == "```python\nprint('hi')\n```"
    assert (blocks[0].start_line, blocks[0].end_line) == (1, 3)


def test_unterminated_fence_runs_to_end_of_file():
    text = "```\nprint(1)\nprint(2)\n"
    blocks = parse_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].kind == "code"
    assert blocks[0].end_line == 3


def test_tilde_fence_is_recognised():
    text = "~~~\ncode here\n~~~\n"
    blocks = parse_blocks(text)
    assert blocks[0].kind == "code"


def test_short_closing_fence_does_not_close_a_longer_opening_fence():
    text = "````\n```\nstill code\n````\n"
    blocks = parse_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].kind == "code"
    assert blocks[0].end_line == 4


def test_heading_inside_fence_is_not_treated_as_a_heading():
    text = "```\n# not a heading\n```\n"
    blocks = parse_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].kind == "code"


def test_pipe_table_with_separator_row():
    text = "| A | B |\n| --- | --- |\n| 1 | 2 |\n"
    blocks = parse_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].kind == "table"
    assert blocks[0].text == text.strip("\n")
    assert (blocks[0].start_line, blocks[0].end_line) == (1, 3)


def test_table_with_alignment_colons():
    text = "| A | B |\n| :--- | ---: |\n| 1 | 2 |\n"
    blocks = parse_blocks(text)
    assert blocks[0].kind == "table"


def test_table_stops_at_first_row_without_a_pipe():
    text = "| A | B |\n| --- | --- |\n| 1 | 2 |\nplain text\n"
    blocks = parse_blocks(text)
    assert _kinds(blocks) == ["table", "paragraph"]
    assert blocks[0].end_line == 3


def test_line_with_pipe_but_no_separator_row_is_not_a_table():
    text = "a | b\nnot a separator\n"
    blocks = parse_blocks(text)
    assert blocks[0].kind == "paragraph"


def test_bullet_list_run():
    text = "- one\n- two\n- three\n"
    blocks = parse_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].kind == "list"
    assert (blocks[0].start_line, blocks[0].end_line) == (1, 3)


def test_ordered_list_run():
    text = "1. first\n2. second\n3. third\n"
    blocks = parse_blocks(text)
    assert blocks[0].kind == "list"


def test_list_keeps_indented_continuation_line():
    text = "- one\n  continued\n- two\n"
    blocks = parse_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].kind == "list"
    assert blocks[0].end_line == 3


def test_three_dashes_alone_is_not_a_bullet_list():
    blocks = parse_blocks("---\n")
    assert blocks[0].kind == "paragraph"


def test_paragraph_runs_until_blank_line():
    text = "Line one\nLine two\nLine three\n"
    blocks = parse_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].kind == "paragraph"
    assert blocks[0].text == text.strip("\n")


def test_paragraph_stops_before_a_following_heading():
    text = "Intro text.\n# Heading\n"
    blocks = parse_blocks(text)
    assert _kinds(blocks) == ["paragraph", "heading"]
    assert blocks[0].text == "Intro text."


def test_paragraph_stops_before_a_following_table():
    text = "Intro\n| A | B |\n| - | - |\n"
    blocks = parse_blocks(text)
    assert _kinds(blocks) == ["paragraph", "table"]
    assert blocks[0].end_line == 1


def test_paragraph_stops_before_a_following_list():
    text = "Intro\n- item\n"
    blocks = parse_blocks(text)
    assert _kinds(blocks) == ["paragraph", "list"]


def test_setext_underline_is_read_as_part_of_the_paragraph():
    text = "Title\n=====\n"
    blocks = parse_blocks(text)
    assert _kinds(blocks) == ["paragraph"]
    assert blocks[0].text == text.strip("\n")


def test_full_document_produces_expected_block_sequence():
    text = (
        "# Runbook\n"
        "\n"
        "Intro paragraph.\n"
        "\n"
        "## Checks\n"
        "\n"
        "- check one\n"
        "- check two\n"
        "\n"
        "| Name | Status |\n"
        "| --- | --- |\n"
        "| a | ok |\n"
        "\n"
        "```bash\n"
        "run.sh\n"
        "```\n"
    )
    blocks = parse_blocks(text)
    assert _kinds(blocks) == [
        "heading",
        "paragraph",
        "heading",
        "list",
        "table",
        "code",
    ]
