# Local Kakugo 3B Chat

This runbook starts an interactive local chat with
[`ptrdvn/kakugo-3B-kin`](https://huggingface.co/ptrdvn/kakugo-3B-kin) on an
Apple-silicon Mac. It uses MLX and does not start the rejected Qwen adapter.

## What Is Already Ready On This Mac

- Apple M5 with about 32 GB unified memory.
- The pinned Kakugo revision is already in the Hugging Face cache.
- Cached model size: about 6.3 GB.
- Revision: `af4e61e47bdd026a73d9a37814f2df8d92f98013`.

In the setup smoke test, the model loaded in about 4.7 seconds, generated about
15.6 tokens per second, and used about 6.9 GB peak memory. Speed and memory vary
with prompt length and conversation history.

## Start Chatting

From the repository root:

```bash
bash scripts/local/chat_kakugo.sh
```

The launcher installs the pinned MLX dependency into an isolated environment
under `~/.cache/kinyalm/kakugo-3b-kin`, loads the model once, and keeps it in
memory while you chat. The isolation keeps MLX's Transformers 5 dependency
separate from the Transformers 4 training environment used on Lambda. The first
run on another Mac also downloads about 6.3 GB from Hugging Face.

Useful in-chat commands:

| Command | Action |
| --- | --- |
| `/reset` | Clear conversation history |
| `/reasoning on` | Enable Kakugo's experimental reasoning prompt and clear history |
| `/reasoning off` | Return to the model card's concise mode and clear history |
| `/show-reasoning` | Show or hide extracted `<think>` text |
| `/help` | Show the command summary |
| `/quit` | Release the model and exit |

The default display removes `<think>` blocks and stores only the visible answer
in later chat history. This prevents hidden reasoning from being fed back into
the next turn.

## Run One Prompt

Use `--prompt` to load the model, answer once, and exit:

```bash
bash scripts/local/chat_kakugo.sh \
  --prompt 'Muraho! Witwa nde? Subiza mu nteruro imwe.'
```

Now that the model is cached, force a no-network run with:

```bash
bash scripts/local/chat_kakugo.sh --offline
```

For a longer reasoning attempt:

```bash
bash scripts/local/chat_kakugo.sh \
  --reasoning \
  --show-reasoning \
  --max-tokens 768
```

The default is deterministic decoding with a `1.05` repetition penalty, which
matches the model author's recommendation. Six recent conversation turns are
retained by default. Override that with `--history-turns` when needed.

## Important Limitation

Kakugo is a useful Kinyarwanda-specific research baseline, not a validated
tutor. Our initial probes found incorrect definitions, grammar explanations,
and translations. Treat every response as an experiment and have fluent
speakers verify anything that will influence training or evaluation. The
evidence is recorded in the
[Track 2 experiment report](experiments/2026-07-20-track2-baseline-report.md).
