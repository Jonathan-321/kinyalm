"""Tokenizer evaluation metrics that do not depend on a specific tokenizer."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
import re


WORD_RE = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", re.UNICODE)


@dataclass(frozen=True)
class TokenizationStats:
    """Compact stats for one text example."""

    text: str
    word_count: int
    character_count: int
    token_count: int

    @property
    def tokens_per_word(self) -> float:
        if self.word_count == 0:
            return 0.0
        return self.token_count / self.word_count

    @property
    def tokens_per_character(self) -> float:
        if self.character_count == 0:
            return 0.0
        return self.token_count / self.character_count


def count_words(text: str) -> int:
    """Count word-like units for tokenizer comparison."""

    return len(WORD_RE.findall(text))


def summarize_tokenization(text: str, token_ids: Sequence[int]) -> TokenizationStats:
    """Summarize one tokenizer output."""

    return TokenizationStats(
        text=text,
        word_count=count_words(text),
        character_count=len(text),
        token_count=len(token_ids),
    )


def summarize_many(
    texts: Iterable[str],
    encode: Callable[[str], Sequence[int]],
) -> list[TokenizationStats]:
    """Summarize a tokenizer over many text examples."""

    return [summarize_tokenization(text, encode(text)) for text in texts]


def average_tokens_per_word(stats: Iterable[TokenizationStats]) -> float:
    """Compute corpus-level tokens per word."""

    total_tokens = 0
    total_words = 0
    for item in stats:
        total_tokens += item.token_count
        total_words += item.word_count
    if total_words == 0:
        return 0.0
    return total_tokens / total_words
