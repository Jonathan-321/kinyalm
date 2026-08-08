import sys
from types import ModuleType, SimpleNamespace

from scripts.generate_gemma4_probe import load_model_and_tokenizer


def test_full_precision_cuda_probe_places_model_on_gpu(monkeypatch):
    calls = {}

    torch = ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.float16 = "float16"
    torch.float32 = "float32"
    torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        is_bf16_supported=lambda: True,
        current_device=lambda: 0,
    )

    class FakeTokenizer:
        pad_token_id = 0

    class FakeTokenizerLoader:
        @staticmethod
        def from_pretrained(model, **kwargs):
            calls["tokenizer"] = (model, kwargs)
            return FakeTokenizer()

    class FakeConfigLoader:
        @staticmethod
        def from_pretrained(model, **kwargs):
            calls["config"] = (model, kwargs)
            return SimpleNamespace(model_type="gemma4_unified")

    class FakeModel:
        config = SimpleNamespace(use_cache=False)

        def eval(self):
            calls["eval"] = True

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(model, **kwargs):
            calls["model"] = (model, kwargs)
            return FakeModel()

    transformers = ModuleType("transformers")
    transformers.AutoConfig = FakeConfigLoader
    transformers.AutoModelForCausalLM = FakeModelLoader
    transformers.AutoModelForMultimodalLM = FakeModelLoader
    transformers.AutoTokenizer = FakeTokenizerLoader
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    model, _ = load_model_and_tokenizer(
        SimpleNamespace(
            model="google/gemma-4-12B-it",
            model_revision="immutable-revision",
            no_quant=True,
            adapter=None,
        )
    )

    assert calls["model"][1]["device_map"] == {"": 0}
    assert calls["model"][1]["dtype"] == "bfloat16"
    assert "quantization_config" not in calls["model"][1]
    assert calls["eval"] is True
    assert model.config.use_cache is True
