#!/usr/bin/env python3
"""Run MLX-LM while preserving the default LoRA adapter mapping."""

from __future__ import annotations

from typing import Any


def resolve_model_paths(
    provider: Any,
    model_path: str,
    adapter_path: str | None,
    draft_model_path: str | None,
) -> tuple[str, str | None, str | None]:
    """Resolve aliases before replacing the model alias with its file path."""
    resolved_adapter = provider._adapter_map.get(model_path, adapter_path)
    resolved_model = provider._model_map.get(model_path, model_path)
    resolved_draft = provider._draft_model_map.get(
        draft_model_path, draft_model_path
    )
    return resolved_model, resolved_adapter, resolved_draft


def install_adapter_aware_provider(server: Any) -> None:
    """Install a provider that does not drop the CLI default adapter."""

    class AdapterAwareModelProvider(server.ModelProvider):
        def load(
            self,
            model_path: str,
            adapter_path: str | None = None,
            draft_model_path: str | None = None,
        ) -> tuple[Any, Any]:
            model_key = resolve_model_paths(
                self, model_path, adapter_path, draft_model_path
            )
            if self.model_key != model_key:
                self._load(*model_key)
            return self.model, self.tokenizer

    server.ModelProvider = AdapterAwareModelProvider


def main() -> None:
    from mlx_lm import server

    install_adapter_aware_provider(server)
    server.main()


if __name__ == "__main__":
    main()
