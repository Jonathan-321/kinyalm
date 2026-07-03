from kinyalm.tokenization.metrics import (
    average_tokens_per_word,
    count_words,
    summarize_many,
    summarize_tokenization,
)


def test_count_words_handles_basic_kinyarwanda_examples():
    assert count_words("Muraho! Amakuru?") == 2
    assert count_words("Ndi umunyeshuri.") == 2
    assert count_words("Kigali-Rwanda") == 1


def test_summarize_tokenization_counts_tokens_words_and_chars():
    stats = summarize_tokenization("Muraho!", [10, 11, 12])

    assert stats.word_count == 1
    assert stats.character_count == 7
    assert stats.token_count == 3
    assert stats.tokens_per_word == 3.0


def test_summarize_many_uses_supplied_encoder():
    def character_encoder(text: str) -> list[int]:
        return list(range(len(text)))

    stats = summarize_many(["Muraho", "Amakuru"], character_encoder)

    assert [item.token_count for item in stats] == [6, 7]
    assert average_tokens_per_word(stats) == 13 / 2
