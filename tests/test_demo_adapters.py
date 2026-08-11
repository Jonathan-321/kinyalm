from kinyalm.demo.adapters import AdapterComparisonRuntime


class FakeLoraLayer:
    def __init__(self, scale):
        self.lora_a = object()
        self.lora_b = object()
        self.scale = scale


class FakeModel:
    def __init__(self, layers):
        self.layers = layers

    def named_modules(self):
        return [(f"layer.{index}", layer) for index, layer in enumerate(self.layers)]


class FakeAdapterRuntime:
    def __init__(self):
        self.layers = [FakeLoraLayer(1.0), FakeLoraLayer(0.5)]
        self.model = FakeModel(self.layers)
        self.cache_key = ("checkpoint", "adapter")
        self.calls = []
        self.closed = False

    def generate_messages(self, **kwargs):
        self.calls.append(
            {
                "scales": [layer.scale for layer in self.layers],
                "cache_key": self.cache_key,
                "kwargs": kwargs,
            }
        )
        return {"response": "answer"}

    def close(self):
        self.closed = True


def test_comparison_runtime_disables_only_lora_contribution_for_base():
    inner = FakeAdapterRuntime()
    runtime = AdapterComparisonRuntime(inner)

    result = runtime.generate_variant_messages(
        variant="base",
        messages=[{"role": "user", "content": "Muraho"}],
        max_new_tokens=12,
        enable_thinking=False,
    )

    assert result["runtime_variant"] == "base"
    assert inner.calls[0]["scales"] == [0.0, 0.0]
    assert inner.calls[0]["cache_key"][-1] == "base"
    assert [layer.scale for layer in inner.layers] == [1.0, 0.5]
    assert inner.cache_key == ("checkpoint", "adapter")


def test_comparison_runtime_uses_adapter_scales_and_distinct_cache():
    inner = FakeAdapterRuntime()
    runtime = AdapterComparisonRuntime(inner)

    runtime.generate_messages(
        messages=[{"role": "user", "content": "Muraho"}],
        max_new_tokens=12,
        enable_thinking=False,
    )

    assert inner.calls[0]["scales"] == [1.0, 0.5]
    assert inner.calls[0]["cache_key"][-1] == "targeted"
    runtime.close()
    assert inner.closed is True
