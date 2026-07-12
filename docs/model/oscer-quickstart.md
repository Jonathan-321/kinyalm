# OSCER Quickstart

This is the practical path for using OU OSCER/Sooner with this repo.

## Current Login Blocker

The pasted SSH output reached OSCER, accepted the host key, then failed
authentication after three password prompts:

```text
Permission denied (publickey,gssapi-keyex,gssapi-with-mic,keyboard-interactive).
```

That is not a SLURM or GPU problem yet. It means OSCER did not accept the
credentials for `jonathanmuhire@sooner.oscer.ou.edu`.

Try these in order:

```bash
ssh jonathanmuhire@sooner.oscer.ou.edu
ssh jonathanmuhire@sooner2.oscer.ou.edu
```

If both reject the password, reset or verify the OSCER password at:

```text
https://account.oscer.ou.edu/
```

If the account username is not exactly `jonathanmuhire`, use the OSCER username
instead:

```bash
ssh OSCER_USERNAME@sooner.oscer.ou.edu
```

The post-quantum warning is from your newer OpenSSH client noticing that OSCER's
server did not negotiate a post-quantum key exchange. It is not why login failed.

Do not use VS Code Remote SSH on OSCER login nodes. OSCER explicitly warns that
VS Code server processes can overload login nodes.

## First Commands After Login

Run these before submitting real work:

```bash
hostname
pwd
id
sinfo
squeue -u "$USER"
```

Storage model:

- `/home/$USER` is long-term personal storage, typically 20 GB and backed up.
- `/scratch/$USER` is for large temporary job data and outputs. Treat it as
  disposable.

For this project, keep the repo and large outputs under scratch:

```bash
mkdir -p /scratch/$USER/kinyalm
cd /scratch/$USER/kinyalm
```

## Copy This Repo To OSCER

From the Mac, after login works:

```bash
rsync -av \
  --exclude '.git/' \
  --exclude '.pytest_cache/' \
  --exclude 'data/raw/' \
  --exclude 'data/interim/' \
  --exclude 'data/processed/' \
  --exclude 'experiments/' \
  --exclude 'logs/' \
  /Users/jonathanmuhire/Documents/RL/ \
  jonathanmuhire@sooner2.oscer.ou.edu:/scratch/jonathanmuhire/kinyalm/
```

Use `dtn2.oscer.ou.edu` for large file transfers once your account can reach it.

## CPU Smoke Job

On OSCER:

```bash
cd /scratch/$USER/kinyalm
mkdir -p slurm_logs
sbatch scripts/hpc/oscer_project_smoke.sbatch
squeue -u "$USER"
```

When the job finishes, check:

```bash
ls -lh slurm_logs/
tail -n 80 slurm_logs/kinyalm-smoke_*_stdout.txt
tail -n 80 slurm_logs/kinyalm-smoke_*_stderr.txt
```

This only checks that Python can see the repo and that the starter tokenizer
tests pass.

## GPU Smoke Job

After the CPU smoke job passes:

```bash
cd /scratch/$USER/kinyalm
mkdir -p slurm_logs
sbatch scripts/hpc/oscer_gpu_smoke.sbatch
squeue -u "$USER"
```

This checks whether a GPU job starts, whether Python can import `torch`, and
whether CUDA is visible inside the job.

## Resource Choices

Use the smallest partition that proves the next claim:

- `sooner_test`: CPU checks, preprocessing, tokenizer experiments, small
  non-GPU jobs.
- `sooner_gpu_test`: first GPU test when you only need "any GPU".
- `sooner_gpu_test_h100`, `sooner_gpu_test_ada`, or related GPU partitions:
  request a specific GPU class only after the generic GPU job works.
- `sooner_test_longjobs`: only for jobs that truly need 2-7 days.

For this repo's current milestone, the right first OSCER target is not a large
LM run. It is:

```text
repo copied to scratch -> CPU smoke job passes -> GPU smoke job passes ->
tiny training command with logs and checkpoint location defined
```
