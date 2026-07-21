import pytest

from scripts.prepare_local_mlx import build_mlx_config, convert_peft_key


def peft_config():
    return {
        "peft_type": "LORA",
        "base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
        "bias": "none",
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "r": 16,
        "target_modules": ["q_proj", "v_proj"],
        "use_dora": False,
        "use_qalora": False,
        "use_rslora": False,
    }


def test_convert_peft_key_to_mlx_layout():
    converted = convert_peft_key(
        "base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight"
    )

    assert converted == (
        "model.layers.3.self_attn.q_proj.lora_a",
        3,
        "self_attn.q_proj",
    )


def test_build_mlx_config_uses_peft_scaling_and_all_layers():
    keys = []
    for layer in range(2):
        for module in ("self_attn.q_proj", "self_attn.v_proj"):
            for matrix in ("a", "b"):
                key = f"model.layers.{layer}.{module}.lora_{matrix}"
                keys.append((key, layer, module))

    config = build_mlx_config(peft_config(), keys)

    assert config["num_layers"] == 2
    assert config["lora_parameters"] == {
        "keys": ["self_attn.q_proj", "self_attn.v_proj"],
        "rank": 16,
        "scale": 2.0,
        "dropout": 0.0,
    }


def test_build_mlx_config_rejects_missing_middle_layer():
    keys = [
        ("model.layers.0.self_attn.q_proj.lora_a", 0, "self_attn.q_proj"),
        ("model.layers.2.self_attn.q_proj.lora_a", 2, "self_attn.q_proj"),
    ]

    with pytest.raises(ValueError, match="every model layer"):
        build_mlx_config(peft_config(), keys)


def test_build_mlx_config_rejects_inconsistent_modules_between_layers():
    keys = [
        ("model.layers.0.self_attn.q_proj.lora_a", 0, "self_attn.q_proj"),
        ("model.layers.0.self_attn.v_proj.lora_a", 0, "self_attn.v_proj"),
        ("model.layers.1.self_attn.q_proj.lora_a", 1, "self_attn.q_proj"),
    ]

    with pytest.raises(ValueError, match="same LoRA modules"):
        build_mlx_config(peft_config(), keys)
