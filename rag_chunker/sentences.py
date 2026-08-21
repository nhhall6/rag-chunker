"""Sentence splitting for paragraph text, with an abbreviation guard.

Splitting naively on ". " breaks abbreviations like "e.g." or "Dr. Chen"
into fragments that make poor chunk boundaries. This treats a period,
question mark or exclamation mark as a sentence end only when it is
followed by whitespace (or the end of the text) and the word right
before it is not a known abbreviation.
"""

import re

# Common abbreviations that end in a period but do not end a sentence.
# Stored without the trailing dot (internal dots, as in "e.g", are kept)
# and compared case-insensitively.
_ABBREVIATIONS = frozenset({
    "e.g", "i.e", "etc", "vs", "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
    "st", "vol", "fig", "no", "approx", "cf", "al", "ca", "inc", "ltd",
    "co", "corp", "pp", "ed",
})

# A run of sentence-ending punctuation, optionally followed by a closing
# quote or bracket, is a boundary only if whitespace or the end of the
# text follows it. A period stuck between two characters, as in "v1.4",
# never matches this at all.
_BOUNDARY_RE = re.compile(r'[.!?]+[\'")\]]*(?=\s|$)')

# The word right before a boundary, including internal dots so "e.g."
# is captured whole as "e.g" rather than just "g".
_WORD_BEFORE_RE = re.compile(r'[A-Za-z]+(?:\.[A-Za-z]+)*$')


def split_sentences(text):
    """Split ``text`` into a list of sentences.

    Embedded whitespace, including newlines from wrapped source lines,
    is never itself a boundary; only sentence-ending punctuation is.
    A trailing fragment with no terminal punctuation is kept as its
    own sentence.
    """
    if not text or not text.strip():
        return []

    sentences = []
    start = 0

    for match in _BOUNDARY_RE.finditer(text):
        boundary_end = match.end()
        word_match = _WORD_BEFORE_RE.search(text[start:match.start()])
        if word_match and word_match.group().lower() in _ABBREVIATIONS:
            continue

        sentence = text[start:boundary_end].strip()
        if sentence:
            sentences.append(sentence)
        start = boundary_end

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)

    return sentences
