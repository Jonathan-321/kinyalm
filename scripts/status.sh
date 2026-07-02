#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Workspace: $ROOT"
echo
echo "Root repo:"
git -C "$ROOT" status --short --branch
echo
echo "CS336 submodules:"
git -C "$ROOT" submodule status
echo
echo "Project docs:"
echo "  $ROOT/README.md"
echo "  $ROOT/TASKS.md"
echo "  $ROOT/docs/project/start-plan.md"
echo "  $ROOT/docs/project/constraints-and-risks.md"
echo "  $ROOT/docs/team/roles.md"
echo
echo "First implementation target:"
echo "  Kinyarwanda tokenizer analysis on a small, documented corpus"
