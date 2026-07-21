# Local Apple-Silicon KinyaLM Run

This runbook serves the experimental Track 2 Qwen adapter on an Apple-silicon
Mac. It uses a pinned 4-bit MLX conversion of Qwen2.5-7B-Instruct and converts
the uploaded Hugging Face PEFT adapter into MLX's LoRA layout.

## Requirements and Size

- Apple-silicon Mac running macOS.
- Python 3.11 or newer.
- A Hugging Face login with read access to the private KinyaLM model repo.
- About 5 GB of free disk space.
- 16 GB unified memory minimum; 32 GB is comfortable for this 7B model.

The pinned MLX base snapshot is 4.28 GB. The trained adapter is 80.8 MB before
conversion, and the Python runtime adds several hundred megabytes. Hugging Face
caches the model under `~/.cache/huggingface`; the KinyaLM runtime and converted
adapter live under `~/.cache/kinyalm/track2-baseline-a`.

The launcher currently pins MLX-LM `0.31.3`, which includes the upstream fix
for thread-local generation streams used by the HTTP server.

MLX reports Qwen2.5 as the model because that remains the underlying
architecture. The KinyaLM LoRA is a small adapter loaded on top of it, not a
separate full model. The local entrypoint also preserves the adapter when
MLX resolves its `default_model` alias; clients do not need to send an adapter
path with every request.

## Start the Endpoint

From the repository root, run:

```bash
bash scripts/local/start_mlx.sh
```

The first start creates an isolated environment, downloads the pinned assets,
and can take several minutes. Later starts reuse the local files. The launcher
installs a user-level macOS service that keeps running after the launching
terminal closes, restarts after an unexpected failure, and listens only on this
Mac at:

```text
http://127.0.0.1:8080/v1/chat/completions
```

This is a POST API endpoint, not a web page. Opening it directly in a browser
uses GET and returns 404; use the request below or the terminal chat client.

Test one request from another terminal:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "user", "content": "Sobanura ijambo ubupfura mu nteruro ebyiri."}
    ],
    "temperature": 0.2,
    "max_tokens": 256
  }'
```

Or use the terminal chat client:

```bash
python3 scripts/local/chat_mlx.py
```

Check or stop the service with:

```bash
bash scripts/local/status_mlx.sh
bash scripts/local/stop_mlx.sh
```

For foreground logs during development, use `serve_mlx.sh` directly and stop
it with `Ctrl-C`.

## Reproducibility and Limits

The launcher pins the base conversion, adapter commit, and MLX-LM version. It
binds to `127.0.0.1`, so it is not exposed to the network. Override
`KINYALM_PORT` only when port 8080 is already occupied.

The adapter conversion transposes and renames the LoRA matrices without
changing their values. Local outputs can still differ slightly from Lambda
because MLX and bitsandbytes use different 4-bit implementations.

This adapter is for inspection, not release: the first automated generation
check and manual conversations found serious language-quality failures. The
Qwen candidate is now retired from future model selection. Preserve this path
only to reproduce the negative baseline and verify local serving. See the
[complete experiment report](experiments/2026-07-20-track2-baseline-report.md)
for screenshots, findings, and the next-base bake-off.
