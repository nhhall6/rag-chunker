"""Token count estimation used for chunk sizing.

No dependency here can be a real tokenizer without pulling in a model's
vocabulary, so this is a character-class heuristic instead: it looks at
what kind of text a run is (word, digits, CJK, punctuation, newline) and
prices each kind separately. See the README for the accuracy this buys.
"""

import re

# Scripts that tokenize close to one codepoint per token in most BPE
# vocabularies, unlike space-delimited Latin text.
_CJK_RANGES = (
    (0x3040, 0x30FF),  # hiragana + katakana
    (0x3400, 0x4DBF),  # CJK extension A
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xAC00, 0xD7A3),  # hangul syllables
    (0xF900, 0xFAFF),  # CJK compatibility ideographs
)

# Matched left to right against everything that isn't a CJK codepoint,
# which is filtered out before this pattern ever sees it.
_TOKEN_RE = re.compile(
    r"(?P<word>[A-Za-z]+(?:'[A-Za-z]+)*)"
    r"|(?P<number>\d[\d,.]*)"
    r"|(?P<newline>\n)"
    r"|(?P<space>[^\S\n]+)"
    r"|(?P<symbol>[^\sA-Za-z0-9])"
)


def _is_cjk(codepoint):
    return any(lo <= codepoint <= hi for lo, hi in _CJK_RANGES)


def _ceil_div(numerator, denominator):
    return -(-numerator // denominator)


def estimate_tokens(text):
    """Estimate a BPE-style token count for ``text``.

    Word runs cost about one token per four characters, the rough
    subword length in English BPE vocabularies. Digit runs cost one
    token per two characters, since numbers tokenize denser than words.
    Each CJK character is its own token. Newlines and punctuation each
    cost a flat token. Whitespace otherwise is free.

    This lands within roughly 10% of a real tokenizer on English prose.
    It is a sizing heuristic, not a tokenizer, and drifts further on
    code, dense symbols, or non-Latin, non-CJK scripts.
    """
    if not text:
        return 0

    total = 0
    pos = 0
    length = len(text)

    while pos < length:
        codepoint = ord(text[pos])
        if _is_cjk(codepoint):
            total += 1
            pos += 1
            continue

        match = _TOKEN_RE.match(text, pos)
        if match is None:
            # Control characters and anything else uncategorized still
            # cost something rather than vanishing from the estimate.
            total += 1
            pos += 1
            continue

        kind = match.lastgroup
        run = match.group()
        if kind == "word":
            total += max(1, _ceil_div(len(run), 4))
        elif kind == "number":
            total += max(1, _ceil_div(len(run), 2))
        elif kind == "newline" or kind == "symbol":
            total += 1
        # "space" runs contribute nothing.

        pos = match.end()

    return total
