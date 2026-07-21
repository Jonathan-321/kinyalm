from scripts.local.mlx_server import resolve_model_paths


class FakeProvider:
    _model_map = {"default_model": "/models/qwen"}
    _adapter_map = {"default_model": "/adapters/kinyalm"}
    _draft_model_map = {"default_model": None}


def test_default_model_keeps_its_adapter():
    paths = resolve_model_paths(
        FakeProvider(), "default_model", None, "default_model"
    )

    assert paths == ("/models/qwen", "/adapters/kinyalm", None)


def test_explicit_adapter_is_preserved_for_non_default_model():
    paths = resolve_model_paths(
        FakeProvider(), "/models/other", "/adapters/other", None
    )

    assert paths == ("/models/other", "/adapters/other", None)
