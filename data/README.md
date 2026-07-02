# Data Directory

This folder is intentionally split by data stage.

```text
raw/        original data exactly as received or downloaded
interim/    cleaned/intermediate data
processed/  final prepared datasets or small shareable samples
external/   third-party reference files
```

Large data is not committed by default.

Before adding data, record the source in:

```text
docs/data/source-log.md
```

If license or permission is unclear, do not use the data for training.
