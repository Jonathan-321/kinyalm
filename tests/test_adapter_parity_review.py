import json
import subprocess
import sys
from pathlib import Path

import pytest

from kinyalm.evaluation import load_bakeoff_config
from scripts.build_adapter_parity_review import load_parity_results
from scripts.run_multilingual_bakeoff import load_held_out_tasks

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "evaluation" / "gemma4_bakeoff.json"


def write_results(path, candidate_id, tasks, config, *, adapter_id=None):
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(
                json.dumps(
                    {
                        "candidate_id": candidate_id,
                        "model_id": "google/gemma-4-12B-it",
                        "model_revision": "a" * 40,
                        "adapter_id": adapter_id,
                        "adapter_revision": "b" * 40 if adapter_id else None,
                        "inference_backend": "transformers",
                        "quantization": None,
                        "task_id": task.id,
                        "prompt": task.prompt,
                        "system_prompt": config.system_prompt,
                        "seed": config.seed,
                        "max_new_tokens": config.max_new_tokens,
                        "enable_thinking": config.enable_thinking,
                        "status": "ok",
                        "response": f"{candidate_id}: {task.id}",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def test_parity_loader_combines_complete_identical_runs(tmp_path):
    config = load_bakeoff_config(CONFIG)
    _, all_tasks = load_held_out_tasks(config)
    tasks = all_tasks[:2]
    base_path = tmp_path / "base.jsonl"
    adapter_path = tmp_path / "adapter.jsonl"
    write_results(base_path, "base", tasks, config)
    write_results(adapter_path, "peft", tasks, config, adapter_id="kinyalm/adapter")

    candidates = load_parity_results(
        [base_path, adapter_path],
        tasks,
        config,
    )

    assert sorted(candidates) == ["base", "peft"]
    assert set(candidates["peft"]) == {task.id for task in tasks}


def test_parity_loader_rejects_changed_generation_inputs(tmp_path):
    config = load_bakeoff_config(CONFIG)
    _, all_tasks = load_held_out_tasks(config)
    tasks = all_tasks[:1]
    base_path = tmp_path / "base.jsonl"
    adapter_path = tmp_path / "adapter.jsonl"
    write_results(base_path, "base", tasks, config)
    write_results(adapter_path, "peft", tasks, config, adapter_id="kinyalm/adapter")
    row = json.loads(adapter_path.read_text(encoding="utf-8"))
    row["max_new_tokens"] = config.max_new_tokens - 1
    adapter_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="differs in max_new_tokens"):
        load_parity_results([base_path, adapter_path], tasks, config)


def test_parity_review_cli_runs_from_repo_root(tmp_path):
    config = load_bakeoff_config(CONFIG)
    _, all_tasks = load_held_out_tasks(config)
    tasks = all_tasks[:1]
    base_path = tmp_path / "base.jsonl"
    adapter_path = tmp_path / "adapter.jsonl"
    output_dir = tmp_path / "review"
    write_results(base_path, "base", tasks, config)
    write_results(adapter_path, "peft", tasks, config, adapter_id="kinyalm/adapter")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_adapter_parity_review.py",
            "--limit",
            "1",
            "--result",
            str(base_path),
            "--result",
            str(adapter_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "blind-review.csv").is_file()
    assert (output_dir / "blind-key.json").is_file()
    assert (output_dir / "parity-review-manifest.json").is_file()
