#!/usr/bin/env python3
"""Run unchanged Hugging Face models on the held-out KinyaLM task bank."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinyalm.evaluation import (  # noqa: E402
    BakeoffConfig,
    CandidateSpec,
    MlxRuntimeSpec,
    TutorTask,
    append_result,
    benchmark_tasks,
    latest_results,
    load_bakeoff_config,
    load_task_bank,
    write_blind_review_pack,
)

DEFAULT_CONFIG = ROOT / "configs" / "evaluation" / "gemma4_bakeoff.json"


@dataclass(frozen=True)
class RuntimeCandidate:
    """The exact checkpoint and backend used for one evaluation run."""

    id: str
    model_id: str
    revision: str
    source_id: str
    source_model_id: str
    source_revision: str
    backend: str
    backend_version: str | None
    model_type_override: str | None
    ignored_weight_prefixes: tuple[str, ...]
    suppress_token_ids: tuple[int, ...]
    quantization: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pinned base models on held-out tutor prompts."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("transformers", "mlx"),
        default="transformers",
        help="Inference backend (default: transformers)",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        help="Candidate ID to run; repeat to select multiple (default: all)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only model files already present in the Hugging Face cache",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the experiment without importing or loading a model",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        help="Held-out task ID to run; repeat to select multiple",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N selected tasks (for smoke testing)",
    )
    return parser.parse_args()


def select_candidates(
    config: BakeoffConfig, requested_ids: list[str] | None
) -> list[CandidateSpec]:
    """Return requested candidates in configuration order."""

    if not requested_ids:
        return list(config.candidates)
    duplicates = sorted(
        candidate_id
        for candidate_id in set(requested_ids)
        if requested_ids.count(candidate_id) > 1
    )
    if duplicates:
        raise ValueError(f"candidate requested more than once: {', '.join(duplicates)}")

    requested = set(requested_ids)
    known = {candidate.id for candidate in config.candidates}
    unknown = sorted(requested.difference(known))
    if unknown:
        raise ValueError(f"unknown candidate ID: {', '.join(unknown)}")
    return [candidate for candidate in config.candidates if candidate.id in requested]


def load_held_out_tasks(config: BakeoffConfig) -> tuple[Path, list[TutorTask]]:
    """Load the configured task bank and enforce its held-out count."""

    task_bank_path = ROOT / config.task_bank
    tasks = benchmark_tasks(load_task_bank(task_bank_path))
    if len(tasks) != config.expected_task_count:
        raise ValueError(
            f"expected {config.expected_task_count} benchmark-only tasks, "
            f"found {len(tasks)}"
        )
    if {task.split for task in tasks} != {config.task_split}:
        raise ValueError("selected tasks do not match the configured held-out split")
    return task_bank_path, tasks


def select_tasks(
    tasks: list[TutorTask], requested_ids: list[str] | None, limit: int | None
) -> list[TutorTask]:
    """Select a deterministic held-out subset for smoke tests or resumes."""

    selected = list(tasks)
    if requested_ids:
        duplicates = sorted(
            task_id
            for task_id in set(requested_ids)
            if requested_ids.count(task_id) > 1
        )
        if duplicates:
            raise ValueError(f"task requested more than once: {', '.join(duplicates)}")
        requested = set(requested_ids)
        known = {task.id for task in tasks}
        unknown = sorted(requested.difference(known))
        if unknown:
            raise ValueError(f"unknown held-out task ID: {', '.join(unknown)}")
        selected = [task for task in tasks if task.id in requested]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        selected = selected[:limit]
    return selected


def resolve_runtime_candidates(
    config: BakeoffConfig,
    requested_ids: list[str] | None,
    backend: str,
) -> list[RuntimeCandidate]:
    """Resolve source candidates to exact backend-specific checkpoints."""

    if backend == "mlx" and not requested_ids:
        selected = [
            candidate for candidate in config.candidates if candidate.local_mlx
        ]
    else:
        selected = select_candidates(config, requested_ids)

    runtimes: list[RuntimeCandidate] = []
    for candidate in selected:
        if backend == "transformers":
            runtimes.append(
                RuntimeCandidate(
                    id=candidate.id,
                    model_id=candidate.model_id,
                    revision=candidate.revision,
                    source_id=candidate.id,
                    source_model_id=candidate.model_id,
                    source_revision=candidate.revision,
                    backend=backend,
                    backend_version=_package_version("transformers"),
                    model_type_override=None,
                    ignored_weight_prefixes=(),
                    suppress_token_ids=(),
                    quantization=None,
                )
            )
            continue

        local = candidate.local_mlx
        if local is None:
            raise ValueError(f"{candidate.id} has no pinned local MLX runtime")
        runtimes.append(_mlx_runtime_candidate(candidate, local))

    if not runtimes:
        raise ValueError(f"no candidates support the {backend} backend")
    return runtimes


def _mlx_runtime_candidate(
    source: CandidateSpec, local: MlxRuntimeSpec
) -> RuntimeCandidate:
    return RuntimeCandidate(
        id=local.id,
        model_id=local.model_id,
        revision=local.revision,
        source_id=source.id,
        source_model_id=source.model_id,
        source_revision=source.revision,
        backend="mlx",
        backend_version=local.mlx_lm_version,
        model_type_override=local.model_type,
        ignored_weight_prefixes=local.ignored_weight_prefixes,
        suppress_token_ids=local.suppress_token_ids,
        quantization=local.quantization,
    )


class TransformersGenerator:
    """One loaded Gemma candidate and its text-generation processor."""

    def __init__(self, candidate: RuntimeCandidate, seed: int) -> None:
        import torch
        from transformers import (
            AutoModelForMultimodalLM,
            AutoProcessor,
            set_seed,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the Gemma 4 bake-off")

        set_seed(seed)
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            candidate.model_id,
            revision=candidate.revision,
        )
        self.model = AutoModelForMultimodalLM.from_pretrained(
            candidate.model_id,
            revision=candidate.revision,
            dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.input_device = self.model.get_input_embeddings().weight.device

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        enable_thinking: bool,
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=enable_thinking,
        )
        input_tokens = int(inputs["input_ids"].shape[-1])
        prefix_ids = inputs["input_ids"][0].detach().cpu()
        inputs = inputs.to(self.input_device)

        for device_index in range(self.torch.cuda.device_count()):
            self.torch.cuda.reset_peak_memory_stats(device_index)
        started = time.perf_counter()
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        elapsed = time.perf_counter() - started
        generated_ids = output_ids[0][input_tokens:].detach().cpu()
        output_tokens = int(generated_ids.shape[-1])
        raw_response = self.processor.decode(
            generated_ids,
            skip_special_tokens=False,
        )
        parsed_response = self.processor.parse_response(
            generated_ids,
            prefix=prefix_ids,
        )
        if not isinstance(parsed_response, dict):
            raise RuntimeError("Gemma response parser did not return one message")
        response = str(parsed_response.get("content", "")).strip()
        thinking = str(parsed_response.get("thinking", "")).strip()
        if not response:
            raise RuntimeError("Gemma generated no visible response content")
        peak_memory = max(
            self.torch.cuda.max_memory_allocated(index)
            for index in range(self.torch.cuda.device_count())
        ) / (1024**3)
        return {
            "response": response,
            "raw_response": raw_response,
            "thinking": thinking,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_seconds": round(elapsed, 4),
            "tokens_per_second": round(output_tokens / elapsed, 4),
            "peak_gpu_memory_gb": round(peak_memory, 4),
            "finish_reason": (
                "length" if output_tokens >= max_new_tokens else "stop"
            ),
        }

    def close(self) -> None:
        del self.model
        del self.processor
        gc.collect()
        self.torch.cuda.empty_cache()


class MlxGenerator:
    """One local Apple-silicon MLX checkpoint and its tokenizer."""

    def __init__(self, candidate: RuntimeCandidate, seed: int) -> None:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("the MLX backend requires an Apple-silicon Mac")

        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        installed = _package_version("mlx-lm")
        if installed != candidate.backend_version:
            raise RuntimeError(
                f"mlx-lm {candidate.backend_version} is required; found {installed}"
            )

        mx.random.seed(seed)
        self.mx = mx
        self.stream_generate = stream_generate
        self.sampler = make_sampler(temp=0.0, top_p=0.0)
        self.logits_processors = make_logits_processors(
            logit_bias={
                token_id: -float("inf")
                for token_id in candidate.suppress_token_ids
            }
        )
        self.model, self.tokenizer = load_text_only_mlx_model(candidate)

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        enable_thinking: bool,
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.generate_messages(
            messages=messages,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
        )

    def generate_messages(
        self,
        *,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        enable_thinking: bool,
    ) -> dict[str, Any]:
        """Generate one response from a complete chat history."""

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        self.mx.reset_peak_memory()
        started = time.perf_counter()
        chunks: list[str] = []
        final = None
        for generation in self.stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=max_new_tokens,
            sampler=self.sampler,
            logits_processors=self.logits_processors,
        ):
            chunks.append(generation.text)
            final = generation
        elapsed = time.perf_counter() - started
        if final is None:
            raise RuntimeError("MLX generated no output tokens")

        raw_response = "".join(chunks)
        response, thinking = parse_gemma4_response(raw_response)
        if not response:
            raise RuntimeError("Gemma generated no visible response content")
        return {
            "response": response,
            "raw_response": raw_response,
            "thinking": thinking,
            "input_tokens": int(final.prompt_tokens),
            "output_tokens": int(final.generation_tokens),
            "latency_seconds": round(elapsed, 4),
            "tokens_per_second": round(float(final.generation_tps), 4),
            "peak_unified_memory_gb": round(float(final.peak_memory), 4),
            "finish_reason": final.finish_reason,
        }

    def close(self) -> None:
        del self.model
        del self.tokenizer
        gc.collect()
        self.mx.clear_cache()


def parse_gemma4_response(raw_response: str) -> tuple[str, str]:
    """Separate Gemma 4 thought-channel text from reviewer-visible content."""

    import re

    thought_pattern = re.compile(
        r"<\|channel>thought\n(?P<thinking>.*?)<channel\|>", re.DOTALL
    )
    thinking = "\n".join(
        match.group("thinking").strip()
        for match in thought_pattern.finditer(raw_response)
        if match.group("thinking").strip()
    )
    response = thought_pattern.sub("", raw_response).strip()
    response = response.removesuffix("<turn|>").removesuffix("<eos>").strip()
    return response, thinking


def filter_ignored_weights(
    weights: dict[str, Any], prefixes: tuple[str, ...]
) -> dict[str, Any]:
    """Remove only explicitly pinned non-text tensors from a checkpoint."""

    matched = {
        prefix for prefix in prefixes if any(key.startswith(prefix) for key in weights)
    }
    missing = sorted(set(prefixes).difference(matched))
    if missing:
        raise ValueError(
            "configured ignored weight prefixes were not present: "
            + ", ".join(missing)
        )
    return {
        key: value
        for key, value in weights.items()
        if not any(key.startswith(prefix) for prefix in prefixes)
    }


def load_text_only_mlx_model(candidate: RuntimeCandidate) -> tuple[Any, Any]:
    """Strictly load Gemma text weights while excluding pinned vision tensors."""

    from huggingface_hub import snapshot_download
    from mlx_lm.models import gemma4
    from mlx_lm.utils import load_model, load_tokenizer

    model_path = Path(
        snapshot_download(
            repo_id=candidate.model_id,
            revision=candidate.revision,
        )
    )
    ignored_prefixes = candidate.ignored_weight_prefixes

    class TextOnlyGemma4(gemma4.Model):
        def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
            filtered = filter_ignored_weights(weights, ignored_prefixes)
            return super().sanitize(filtered)

    model, model_config = load_model(
        model_path,
        strict=True,
        model_config={"model_type": candidate.model_type_override},
        get_model_classes=lambda config: (TextOnlyGemma4, gemma4.ModelArgs),
    )
    tokenizer = load_tokenizer(
        model_path,
        eos_token_ids=model_config.get("eos_token_id"),
    )
    return model, tokenizer


def run_candidate(
    *,
    candidate: RuntimeCandidate,
    config: BakeoffConfig,
    tasks: list[TutorTask],
    output_dir: Path,
) -> bool:
    """Run or resume one candidate, checkpointing every task response."""

    result_path = output_dir / "raw" / f"{candidate.id}.jsonl"
    existing = latest_results(result_path)
    pending = [
        task for task in tasks if existing.get(task.id, {}).get("status") != "ok"
    ]
    if not pending:
        print(f"{candidate.id}: all {len(tasks)} tasks already completed.")
        return True

    print(f"{candidate.id}: loading {candidate.model_id}@{candidate.revision[:12]}...")
    load_started = time.perf_counter()
    try:
        generator_class = (
            TransformersGenerator
            if candidate.backend == "transformers"
            else MlxGenerator
        )
        generator = generator_class(candidate, config.seed)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"{candidate.id}: model load failed: {error}", file=sys.stderr)
        for task in pending:
            append_result(
                result_path,
                _error_record(candidate, config, task, error, phase="model-load"),
            )
        (output_dir / "errors").mkdir(parents=True, exist_ok=True)
        (output_dir / "errors" / f"{candidate.id}-load.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        return False

    load_seconds = time.perf_counter() - load_started
    print(
        f"{candidate.id}: loaded in {load_seconds:.1f}s; "
        f"{len(pending)} tasks pending."
    )
    all_ok = True
    try:
        for index, task in enumerate(pending, start=1):
            print(f"{candidate.id} [{index}/{len(pending)}] {task.id}", flush=True)
            try:
                generation = generator.generate(
                    system_prompt=config.system_prompt,
                    user_prompt=task.prompt,
                    max_new_tokens=config.max_new_tokens,
                    enable_thinking=config.enable_thinking,
                )
                record = {
                    **_base_record(candidate, config, task),
                    "status": "ok",
                    "model_load_seconds": round(load_seconds, 4),
                    **generation,
                }
            except Exception as exc:
                all_ok = False
                error = f"{type(exc).__name__}: {exc}"
                print(f"{candidate.id} {task.id}: {error}", file=sys.stderr)
                record = _error_record(
                    candidate, config, task, error, phase="generation"
                )
            append_result(result_path, record)
    finally:
        generator.close()
    return all_ok


def build_review_outputs(
    *,
    config: BakeoffConfig,
    candidates: list[RuntimeCandidate],
    tasks: list[TutorTask],
    output_dir: Path,
) -> bool:
    """Create the blind review sheet when every candidate has every task."""

    candidate_results = {
        candidate.id: latest_results(output_dir / "raw" / f"{candidate.id}.jsonl")
        for candidate in candidates
    }
    missing = [
        f"{candidate.id}:{task.id}"
        for candidate in candidates
        for task in tasks
        if task.id not in candidate_results[candidate.id]
    ]
    if missing:
        print(
            "Blind review pack not built; missing results: " + ", ".join(missing[:8]),
            file=sys.stderr,
        )
        return False

    row_count, _ = write_blind_review_pack(
        output_csv=output_dir / "review" / "blind-review.csv",
        key_path=output_dir / "private" / "blind-key.json",
        tasks=tasks,
        candidate_results=candidate_results,
        seed=config.seed,
    )
    print(f"Wrote {row_count} blinded review rows.")
    return True


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_bakeoff_config(config_path)
    task_bank_path, all_tasks = load_held_out_tasks(config)
    tasks = select_tasks(all_tasks, args.task_id, args.limit)
    candidates = resolve_runtime_candidates(config, args.candidate, args.backend)

    summary = {
        "run_name": config.run_name,
        "run_kind": (
            "bf16-reference"
            if args.backend == "transformers"
            else "quantized-local-screen"
        ),
        "backend": args.backend,
        "task_count": len(tasks),
        "task_ids": [task.id for task in tasks],
        "candidates": [asdict(candidate) for candidate in candidates],
        "max_new_tokens": config.max_new_tokens,
        "enable_thinking": config.enable_thinking,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(
        output_dir=output_dir,
        config_path=config_path,
        task_bank_path=task_bank_path,
        summary=summary,
    )

    success = True
    for candidate in candidates:
        success = run_candidate(
            candidate=candidate,
            config=config,
            tasks=tasks,
            output_dir=output_dir,
        ) and success

    complete_tasks = {task.id for task in tasks} == {task.id for task in all_tasks}
    if complete_tasks:
        success = build_review_outputs(
            config=config,
            candidates=candidates,
            tasks=tasks,
            output_dir=output_dir,
        ) and success
    else:
        print("Skipped review pack because only a task subset was run.")
    return 0 if success else 1


def _base_record(
    candidate: RuntimeCandidate, config: BakeoffConfig, task: TutorTask
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_name": config.run_name,
        "candidate_id": candidate.id,
        "model_id": candidate.model_id,
        "model_revision": candidate.revision,
        "source_candidate_id": candidate.source_id,
        "source_model_id": candidate.source_model_id,
        "source_model_revision": candidate.source_revision,
        "inference_backend": candidate.backend,
        "inference_backend_version": candidate.backend_version,
        "model_type_override": candidate.model_type_override,
        "ignored_weight_prefixes": candidate.ignored_weight_prefixes,
        "suppress_token_ids": candidate.suppress_token_ids,
        "quantization": candidate.quantization,
        "task_id": task.id,
        "category": task.category,
        "split": task.split,
        "prompt": task.prompt,
        "review_focus": task.review_focus,
        "system_prompt": config.system_prompt,
        "seed": config.seed,
        "max_new_tokens": config.max_new_tokens,
        "enable_thinking": config.enable_thinking,
        "generated_at_utc": _utc_now(),
    }


def _error_record(
    candidate: RuntimeCandidate,
    config: BakeoffConfig,
    task: TutorTask,
    error: str,
    *,
    phase: str,
) -> dict[str, Any]:
    return {
        **_base_record(candidate, config, task),
        "status": "error",
        "error_phase": phase,
        "error": error,
    }


def _write_manifest(
    *,
    output_dir: Path,
    config_path: Path,
    task_bank_path: Path,
    summary: dict[str, Any],
) -> None:
    manifest = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "git_commit": _git_commit(),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "task_bank_path": str(task_bank_path.relative_to(ROOT)),
        "task_bank_sha256": _sha256(task_bank_path),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            package: _package_version(package)
            for package in (
                "torch",
                "transformers",
                "accelerate",
                "huggingface-hub",
                "mlx",
                "mlx-lm",
            )
        },
        **summary,
    }
    path = output_dir / "run-manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("config_sha256") != manifest["config_sha256"]:
            raise ValueError("output directory belongs to a different bake-off config")
        if existing.get("task_bank_sha256") != manifest["task_bank_sha256"]:
            raise ValueError("task bank changed since this bake-off started")
        for key in ("backend", "task_ids", "candidates"):
            current = json.loads(json.dumps(manifest.get(key)))
            if existing.get(key) != current:
                raise ValueError(
                    f"output directory belongs to different {key} settings"
                )
        return
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
