"""Switch one loaded MLX LoRA runtime between base and adapted behavior."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol


class _AdapterRuntime(Protocol):
    model: Any
    cache_key: tuple[Any, ...]

    def generate_messages(
        self,
        *,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        enable_thinking: bool,
        on_text: Callable[[str], None] | None = None,
    ) -> dict[str, Any]: ...

    def close(self) -> None: ...


class AdapterComparisonRuntime:
    """Expose unchanged-base and LoRA outputs without loading two base models."""

    VARIANTS = {"base", "targeted"}

    def __init__(self, runtime: _AdapterRuntime) -> None:
        self.runtime = runtime
        self._cache_key = runtime.cache_key
        self._lora_layers: list[tuple[Any, float]] = []
        for _, module in runtime.model.named_modules():
            if all(hasattr(module, name) for name in ("lora_a", "lora_b", "scale")):
                self._lora_layers.append((module, float(module.scale)))
        if not self._lora_layers:
            raise ValueError("the loaded model does not contain any LoRA layers")

    @property
    def lora_layer_count(self) -> int:
        return len(self._lora_layers)

    def generate_messages(
        self,
        *,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        enable_thinking: bool,
        on_text: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        return self.generate_variant_messages(
            variant="targeted",
            messages=messages,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
            on_text=on_text,
        )

    def generate_variant_messages(
        self,
        *,
        variant: str,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        enable_thinking: bool,
        on_text: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        if variant not in self.VARIANTS:
            raise ValueError(f"unknown runtime variant: {variant}")

        enabled = variant == "targeted"
        for module, scale in self._lora_layers:
            module.scale = scale if enabled else 0.0
        self.runtime.cache_key = (*self._cache_key, variant)
        try:
            result = self.runtime.generate_messages(
                messages=messages,
                max_new_tokens=max_new_tokens,
                enable_thinking=enable_thinking,
                on_text=on_text,
            )
            return {**result, "runtime_variant": variant}
        finally:
            for module, scale in self._lora_layers:
                module.scale = scale
            self.runtime.cache_key = self._cache_key

    def close(self) -> None:
        self.runtime.close()
