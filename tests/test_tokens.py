from rag_chunker import estimate_tokens


def test_empty_string_is_zero_tokens():
    assert estimate_tokens("") == 0


def test_short_word_costs_one_token():
    assert estimate_tokens("hi") == 1


def test_word_cost_is_ceil_of_length_over_four():
    assert estimate_tokens("hello") == 2  # ceil(5 / 4)
    assert estimate_tokens("word") == 1  # ceil(4 / 4)


def test_spaces_between_words_are_free():
    assert estimate_tokens("hello world") == 4  # 2 + 0 + 2


def test_apostrophe_keeps_a_contraction_as_one_word():
    # "don't" is one 5-character word run, not "don" + "'" + "t" split
    # across the symbol class, so it costs ceil(5 / 4), not three tokens.
    assert estimate_tokens("don't") == 2


def test_digit_runs_cost_one_token_per_two_characters():
    assert estimate_tokens("12345") == 3  # ceil(5 / 2)
    assert estimate_tokens("7") == 1


def test_newline_costs_a_flat_token():
    assert estimate_tokens("a\nb") == 3  # 1 + 1 + 1


def test_blank_line_costs_only_the_newline():
    assert estimate_tokens("   \n   ") == 1


def test_punctuation_costs_one_token_per_character():
    assert estimate_tokens("...") == 3
    assert estimate_tokens("a, b.") == 4  # "a" + "," + "b" + "."


def test_cjk_characters_cost_one_token_each():
    assert estimate_tokens("你好") == 2
    assert estimate_tokens("hi 你好") == 3  # "hi" + 2 CJK chars


def test_longer_prose_stays_close_to_a_quarter_of_character_count():
    text = "The quick brown fox jumps over the lazy dog"
    tokens = estimate_tokens(text)
    # Nine short words: each is at most a couple of tokens, well under
    # one token per character and well over one token for the whole line.
    assert 1 < tokens < len(text)


def test_estimate_is_deterministic():
    text = "Retry the job, watch for e.g. rate limits (v1.4)."
    assert estimate_tokens(text) == estimate_tokens(text)
