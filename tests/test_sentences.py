from rag_chunker import split_sentences


def test_empty_string_yields_no_sentences():
    assert split_sentences("") == []


def test_whitespace_only_yields_no_sentences():
    assert split_sentences("   \n  ") == []


def test_single_sentence_with_no_terminal_punctuation():
    assert split_sentences("no ending punctuation here") == [
        "no ending punctuation here"
    ]


def test_splits_on_period_between_sentences():
    assert split_sentences("Retry the job. Then check the logs.") == [
        "Retry the job.",
        "Then check the logs.",
    ]


def test_splits_on_question_and_exclamation_marks():
    assert split_sentences("Did it fail? Yes! Try again.") == [
        "Did it fail?",
        "Yes!",
        "Try again.",
    ]


def test_trailing_fragment_without_terminal_punctuation_is_kept():
    assert split_sentences("First sentence. trailing thought") == [
        "First sentence.",
        "trailing thought",
    ]


def test_abbreviation_eg_does_not_split():
    text = "Use a fallback, e.g. a cached response, when the call fails."
    assert split_sentences(text) == [text]


def test_abbreviation_title_does_not_split():
    text = "Ask Dr. Chen before changing the schema."
    assert split_sentences(text) == [text]


def test_decimal_version_number_is_not_split_mid_number():
    text = "Ship v1.4 once the tests pass. Then tag the release."
    assert split_sentences(text) == [
        "Ship v1.4 once the tests pass.",
        "Then tag the release.",
    ]


def test_multiple_abbreviations_in_one_sentence():
    text = "See the docs (e.g. the README), etc. before asking Dr. Chen."
    assert split_sentences(text) == [text]


def test_newline_wrapped_sentence_stays_one_sentence():
    text = "This is a long\nsentence that wraps across a line break."
    assert split_sentences(text) == [text]


def test_newline_between_sentences_still_splits():
    text = "First sentence.\nSecond sentence."
    assert split_sentences(text) == ["First sentence.", "Second sentence."]


def test_closing_quote_after_punctuation_stays_with_sentence():
    text = 'She said "stop." Then she left.'
    assert split_sentences(text) == ['She said "stop."', "Then she left."]
