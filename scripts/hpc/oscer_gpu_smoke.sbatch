#!/bin/bash
#SBATCH --partition=sooner_gpu_test
#SBATCH --container=el9hw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --time=00:20:00
#SBATCH --job-name=kinyalm-gpu-smoke
#SBATCH --output=slurm_logs/kinyalm-gpu-smoke_%J_stdout.txt
#SBATCH --error=slurm_logs/kinyalm-gpu-smoke_%J_stderr.txt

set -euo pipefail

mkdir -p slurm_logs

echo "Job started at: $(date)"
echo "Running on: $(hostname)"
echo "Working directory: $(pwd)"

module purge || true
module load Python/3.13.5-GCCcore-14.3.0 || true

python3 --version
nvidia-smi || true

python3 - <<'PY'
try:
    import torch
except Exception as exc:
    raise SystemExit(f"torch import failed: {exc}")

print(f"torch: {torch.__version__}")
print(f"cuda_available: {torch.cuda.is_available()}")
print(f"cuda_device_count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones((1024, 1024), device=device)
    y = x @ x
    print(f"device_name: {torch.cuda.get_device_name(0)}")
    print(f"matrix_sum: {float(y.sum().cpu())}")
else:
    raise SystemExit("CUDA is not visible inside this job.")
PY

echo "Job finished at: $(date)"
