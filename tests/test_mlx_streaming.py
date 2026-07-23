from dataclasses import dataclass

from scripts.run_multilingual_bakeoff import MlxGenerator


class FakeTokenizer:
    bos_token = "<bos>"

    def apply_chat_template(self, messages, **kwargs):
        assert messages[-1]["role"] == "user"
        assert kwargs["enable_thinking"] is False
        return "formatted prompt"

    def encode(self, prompt, add_special_tokens):
        assert prompt == "formatted prompt"
        assert add_special_tokens is True
        return [1, 2, 3]


class FakeMx:
    def reset_peak_memory(self):
        return None


@dataclass
class FakeGeneration:
    text: str
    token: int = 4
    prompt_tokens: int = 20
    prompt_tps: float = 100.0
    generation_tokens: int = 2
    generation_tps: float = 8.0
    peak_memory: float = 11.0
    finish_reason: str = "stop"


class FakePromptCache:
    def __init__(self):
        self.inserted = None

    def fetch_nearest_cache(self, key, prompt):
        assert key == "fake-model"
        return None, prompt

    def insert_cache(self, key, tokens, cache):
        self.inserted = (key, tokens, cache)


def test_mlx_generator_forwards_text_as_it_is_generated():
    generator = MlxGenerator.__new__(MlxGenerator)
    generator.tokenizer = FakeTokenizer()
    generator.mx = FakeMx()
    generator.model = object()
    generator.sampler = object()
    generator.logits_processors = []
    generator.make_prompt_cache = lambda model: ["empty-cache"]
    generator.prompt_cache = FakePromptCache()
    generator.cache_key = "fake-model"
    generator.stream_generate = lambda *args, **kwargs: iter(
        [FakeGeneration("Muraho "), FakeGeneration("neza.")]
    )
    chunks = []

    result = generator.generate_messages(
        messages=[{"role": "user", "content": "Hi"}],
        max_new_tokens=32,
        enable_thinking=False,
        on_text=chunks.append,
    )

    assert chunks == ["Muraho ", "neza."]
    assert result["response"] == "Muraho neza."
    assert result["prompt_tokens_per_second"] == 100.0
    assert result["cached_input_tokens"] == 0
    assert generator.prompt_cache.inserted[1] == [1, 2, 3, 4, 4]
    assert result["first_token_seconds"] >= 0
