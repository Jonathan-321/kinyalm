#!/bin/bash
#SBATCH --partition=sooner_test
#SBATCH --container=el9hw
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=00:15:00
#SBATCH --job-name=kinyalm-smoke
#SBATCH --output=slurm_logs/kinyalm-smoke_%J_stdout.txt
#SBATCH --error=slurm_logs/kinyalm-smoke_%J_stderr.txt

set -euo pipefail

mkdir -p slurm_logs

echo "Job started at: $(date)"
echo "Running on: $(hostname)"
echo "Working directory: $(pwd)"

module purge || true
module load Python/3.13.5-GCCcore-14.3.0 || true

python3 --version
python3 scripts/check_project.py

if python3 -c "import pytest" >/dev/null 2>&1; then
  PYTHONPATH=src python3 -m pytest tests/test_tokenizer_metrics.py -q
else
  echo "pytest is not installed in this Python environment; skipping pytest."
  echo "Install project dependencies or load a Python environment with pytest."
fi

echo "Job finished at: $(date)"
