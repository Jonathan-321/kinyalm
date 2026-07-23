#!/usr/bin/env python3
"""Prepare the experimental Track 2 adapter for local MLX inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

DEFAULT_BASE_REPO = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DEFAULT_BASE_REVISION = "c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed"
DEFAULT_ADAPTER_REPO = "kinyalm/kinyalm-qwen2.5-7b-track2-baseline-a"
DEFAULT_ADAPTER_REVISION = "e370728b6c9f5c0c6df57d450261982a69536b83"

PEFT_LORA_KEY = re.compile(
    r"^(?:base_model\.model\.)?"
    r"(?P<module>model\.layers\.(?P<layer>\d+)\..+)\."
    r"lora_(?P<matrix>[AB])(?:\.[^.]+)?\.weight$"
)
LAYER_MODULE = re.compile(r"^model\.layers\.(?P<layer>\d+)\.(?P<module>.+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_peft_key(key: str) -> tuple[str, int, str]:
    """Return the MLX key, layer index, and layer-relative module name."""
    match = PEFT_LORA_KEY.fullmatch(key)
    if match is None:
        raise ValueError(f"Unsupported PEFT adapter weight: {key}")

    module = match.group("module")
    layer_match = LAYER_MODULE.fullmatch(module)
    if layer_match is None:  # pragma: no cover - guarded by PEFT_LORA_KEY
        raise ValueError(f"Adapter weight is not attached to a model layer: {key}")

    matrix = match.group("matrix").lower()
    return (
        f"{module}.lora_{matrix}",
        int(match.group("layer")),
        layer_match.group("module"),
    )


def validate_peft_config(config: dict[str, Any]) -> None:
    unsupported = []
    if config.get("peft_type") != "LORA":
        unsupported.append("peft_type must be LORA")
    if config.get("bias", "none") != "none":
        unsupported.append("bias adapters")
    if config.get("modules_to_save"):
        unsupported.append("modules_to_save")
    if config.get("rank_pattern"):
        unsupported.append("rank_pattern")
    if config.get("alpha_pattern"):
        unsupported.append("alpha_pattern")
    if config.get("use_dora"):
        unsupported.append("DoRA")
    if config.get("use_rslora"):
        unsupported.append("RS-LoRA")
    if config.get("use_qalora"):
        unsupported.append("QA-LoRA")
    if unsupported:
        raise ValueError(
            "This converter cannot preserve: " + ", ".join(unsupported)
        )


def build_mlx_config(
    peft_config: dict[str, Any],
    converted_keys: list[tuple[str, int, str]],
) -> dict[str, Any]:
    validate_peft_config(peft_config)
    if not converted_keys:
        raise ValueError("The PEFT adapter contains no LoRA weights")

    rank = int(peft_config["r"])
    alpha = float(peft_config["lora_alpha"])
    layers = {layer for _, layer, _ in converted_keys}
    expected_layers = set(range(max(layers) + 1))
    if layers != expected_layers:
        raise ValueError("MLX conversion requires LoRA weights for every model layer")

    module_keys = sorted({module for _, _, module in converted_keys})
    modules_by_layer = {
        layer: {
            module
            for _, item_layer, module in converted_keys
            if item_layer == layer
        }
        for layer in expected_layers
    }
    if any(modules != set(module_keys) for modules in modules_by_layer.values()):
        raise ValueError("Every model layer must contain the same LoRA modules")
    return {
        "fine_tune_type": "lora",
        "num_layers": len(expected_layers),
        "lora_parameters": {
            "keys": module_keys,
            "rank": rank,
            "scale": alpha / rank,
            "dropout": 0.0,
        },
        "source": {
            "format": "huggingface-peft",
            "base_model_name_or_path": peft_config.get(
                "base_model_name_or_path"
            ),
            "peft_version": peft_config.get("peft_version"),
        },
    }


def convert_adapter(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    try:
        import mlx.core as mx
    except ImportError as error:  # pragma: no cover - platform/runtime guard
        raise RuntimeError(
            "MLX is required for lossless bfloat16 conversion. "
            "Run scripts/local/serve_mlx.sh on an Apple-silicon Mac."
        ) from error

    config_path = source_dir / "adapter_config.json"
    weights_path = source_dir / "adapter_model.safetensors"
    with config_path.open(encoding="utf-8") as handle:
        peft_config = json.load(handle)

    peft_weights = mx.load(str(weights_path))
    converted_keys: list[tuple[str, int, str]] = []
    mlx_weights = {}
    for source_key, value in peft_weights.items():
        mlx_key, layer, module = convert_peft_key(source_key)
        if mlx_key in mlx_weights:
            raise ValueError(f"Duplicate converted adapter weight: {mlx_key}")
        mlx_weights[mlx_key] = value.T
        converted_keys.append((mlx_key, layer, module))

    expected_pairs = {key.rsplit(".", 1)[0] for key in mlx_weights}
    rank = int(peft_config["r"])
    for prefix in expected_pairs:
        if {f"{prefix}.lora_a", f"{prefix}.lora_b"} - mlx_weights.keys():
            raise ValueError(f"Incomplete LoRA A/B pair for {prefix}")
        lora_a = mlx_weights[f"{prefix}.lora_a"]
        lora_b = mlx_weights[f"{prefix}.lora_b"]
        if lora_a.shape[1] != rank or lora_b.shape[0] != rank:
            raise ValueError(f"LoRA rank does not match adapter config for {prefix}")

    mlx_config = build_mlx_config(peft_config, converted_keys)
    output_dir.mkdir(parents=True, exist_ok=False)
    mx.save_safetensors(
        str(output_dir / "adapters.safetensors"),
        mlx_weights,
        metadata={"format": "mlx", "source_format": "huggingface-peft"},
    )
    (output_dir / "adapter_config.json").write_text(
        json.dumps(mlx_config, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return {
        "source_weight_count": len(peft_weights),
        "converted_weight_count": len(mlx_weights),
        "num_layers": mlx_config["num_layers"],
        "module_keys": mlx_config["lora_parameters"]["keys"],
        "rank": mlx_config["lora_parameters"]["rank"],
        "scale": mlx_config["lora_parameters"]["scale"],
        "source_sha256": sha256_file(weights_path),
        "converted_sha256": sha256_file(output_dir / "adapters.safetensors"),
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    runtime_dir = args.runtime_dir.expanduser().resolve()
    adapter_dir = runtime_dir / "adapter-mlx"
    manifest_path = runtime_dir / "runtime.json"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    base_path = Path(
        snapshot_download(
            repo_id=args.base_repo,
            revision=args.base_revision,
            local_files_only=args.offline,
        )
    ).resolve()
    source_path = Path(
        snapshot_download(
            repo_id=args.adapter_repo,
            revision=args.adapter_revision,
            allow_patterns=["adapter_config.json", "adapter_model.safetensors"],
            local_files_only=args.offline,
        )
    ).resolve()

    previous_manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous_manifest = {}
    previous_adapter = previous_manifest.get("adapter", {})
    previous_conversion = previous_adapter.get("conversion", {})
    source_sha256 = sha256_file(source_path / "adapter_model.safetensors")
    matching_source = (
        previous_adapter.get("repo_id") == args.adapter_repo
        and previous_adapter.get("revision") == args.adapter_revision
        and previous_conversion.get("source_sha256") == source_sha256
    )
    needs_conversion = args.force or not matching_source or not (
        (adapter_dir / "adapter_config.json").is_file()
        and (adapter_dir / "adapters.safetensors").is_file()
    )
    conversion: dict[str, Any]
    if needs_conversion:
        with tempfile.TemporaryDirectory(
            prefix="adapter-mlx-", dir=runtime_dir
        ) as temp_dir:
            temp_output = Path(temp_dir) / "converted"
            conversion = convert_adapter(source_path, temp_output)
            if adapter_dir.exists():
                shutil.rmtree(adapter_dir)
            temp_output.rename(adapter_dir)
    else:
        conversion = dict(previous_conversion)
        conversion["reused"] = True

    manifest = {
        "schema_version": 1,
        "prepared_at": datetime.now(UTC).isoformat(),
        "base": {
            "repo_id": args.base_repo,
            "revision": args.base_revision,
            "path": str(base_path),
        },
        "adapter": {
            "repo_id": args.adapter_repo,
            "revision": args.adapter_revision,
            "source_path": str(source_path),
            "path": str(adapter_dir),
            "conversion": conversion,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the pinned local base and convert the Track 2 adapter."
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=Path("~/.cache/kinyalm/track2-baseline-a"),
    )
    parser.add_argument("--base-repo", default=DEFAULT_BASE_REPO)
    parser.add_argument("--base-revision", default=DEFAULT_BASE_REVISION)
    parser.add_argument("--adapter-repo", default=DEFAULT_ADAPTER_REPO)
    parser.add_argument("--adapter-revision", default=DEFAULT_ADAPTER_REVISION)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main() -> None:
    manifest = prepare(parse_args())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
