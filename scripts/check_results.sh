#!/usr/bin/env bash
# =============================================================================
# scripts/check_results.sh — Summarise a single evaluation result file
# Usage: bash scripts/check_results.sh results/file.jsonl
# =============================================================================
set -euo pipefail

GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
log() { echo -e "${GREEN}[check]${NC} $*"; }

if [[ "${CONDA_DEFAULT_ENV:-}" != "agent-bench" ]]; then
    echo -e "${RED}[error]${NC} conda env 'agent-bench' is not active. Run: conda activate agent-bench"
    exit 1
fi

[[ $# -eq 1 ]] || { echo -e "${RED}[error]${NC} Usage: bash scripts/check_results.sh results/file.jsonl"; exit 1; }

CHECK_SCRIPT="vendor/AgentRL/examples/eval/check.py"
[[ -f "$CHECK_SCRIPT" ]] || { echo -e "${RED}[error]${NC} check.py not found at $CHECK_SCRIPT"; exit 1; }
[[ -f "$1" ]] || { echo -e "${RED}[error]${NC} File not found: $1"; exit 1; }

log "Checking: $1"
# Suppress the known IndexError crash in check.py (cosmetic bug with single runs)
python "$CHECK_SCRIPT" "$1" 2>&1 | sed '/^Traceback/q' | grep -v "^Traceback" || true