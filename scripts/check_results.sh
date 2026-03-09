#!/usr/bin/env bash
# =============================================================================
# scripts/check_results.sh — Summarise evaluation results
# Usage: bash scripts/check_results.sh [results/file.jsonl]
#        bash scripts/check_results.sh          # summarises ALL results files
# =============================================================================
set -euo pipefail

GREEN='\033[0;32m'; NC='\033[0m'
log() { echo -e "${GREEN}[check]${NC} $*"; }

if [[ "${CONDA_DEFAULT_ENV:-}" != "agent-bench" ]]; then
    echo -e "\033[0;31m[error]\033[0m conda env 'agent-bench' is not active. Run: conda activate agent-bench"
    exit 1
fi

CHECK_SCRIPT="vendor/AgentRL/examples/evaluation/check.py"

if [[ $# -eq 1 ]]; then
    log "Checking: $1"
    python "$CHECK_SCRIPT" "$1"
else
    log "Checking all results in ./results/ ..."
    for f in results/*.jsonl; do
        [[ -f "$f" ]] || continue
        echo ""
        echo "━━━ $f"
        python "$CHECK_SCRIPT" "$f"
    done
fi