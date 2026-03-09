#!/usr/bin/env bash
# =============================================================================
# scripts/run_eval.sh — Run AgentBench OS evaluation via AgentRL
#
# Usage:
#   bash scripts/run_eval.sh                          # defaults: gpt-5-mini, os-std
#   bash scripts/run_eval.sh -m gpt-4o -t os-dev -j 16
#   bash scripts/run_eval.sh --resume results/os-std-run1.jsonl
#
# Any extra flags are forwarded directly to server_agent.py, e.g.:
#   bash scripts/run_eval.sh -n 2 --temp 0.0 -v --range 0,20
#
# See vendor/AgentRL/examples/eval/server_agent.py --help for all options.
# Supported tasks: os-std, os-dev
# =============================================================================
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[eval]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
die()  { echo -e "${RED}[error]${NC} $*"; exit 1; }

# ---------------------------------------------------------------------------
# Defaults (override via flags or .env)
# ---------------------------------------------------------------------------
MODEL="${MODEL:-gpt-5-mini}"
BASE_URL="${BASE_URL:-https://api.openai.com/v1}"
TASK="${TASK:-os-std}"
JOBS="${JOBS:-8}"          # concurrent sessions
CONTROLLER="${CONTROLLER:-http://localhost:5020/api}"
RESUME_FILE=""
PASSTHROUGH=()

# Load .env if present
[[ -f .env ]] && export $(grep -v '^#' .env | xargs)

# ---------------------------------------------------------------------------
# Parse flags — known flags set shell vars; everything else is forwarded
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)      MODEL="$2";       shift 2 ;;
        -u|--url)        BASE_URL="$2";    shift 2 ;;
        -t|--task)       TASK="$2";        shift 2 ;;
        -j|--jobs)       JOBS="$2";        shift 2 ;;
        -c|--controller) CONTROLLER="$2";  shift 2 ;;
        --resume)        RESUME_FILE="$2"; shift 2 ;;
        *)               PASSTHROUGH+=("$1"); shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Guard: ensure correct conda env is active
# ---------------------------------------------------------------------------
if [[ "${CONDA_DEFAULT_ENV:-}" != "agent-bench" ]]; then
    die "conda env 'agent-bench' is not active.\n  Run: conda activate agent-bench"
fi

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
# Using server_agent.py directly (stable). agentrl-eval is still experimental
# per https://github.com/THUDM/AgentRL?tab=readme-ov-file#evaluation
EVAL_SCRIPT="vendor/AgentRL/examples/eval/server_agent.py"
[[ -f "$EVAL_SCRIPT" ]] || die "AgentRL eval script not found at $EVAL_SCRIPT. Did you run setup.sh?"

# Check controller is up
# curl -s "$CONTROLLER" >/dev/null 2>&1 \
#     || die "Controller not reachable at $CONTROLLER. Run: bash scripts/start_env.sh --os-only"

# Check API key
if [[ "$BASE_URL" == *"openai"* ]] && [[ -z "${OPENAI_API_KEY:-}" ]]; then
    die "OPENAI_API_KEY is not set. Add it to .env or export it."
fi
if [[ "$BASE_URL" == *"anthropic"* ]] && [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    die "ANTHROPIC_API_KEY is not set. Add it to .env or export it."
fi

# ---------------------------------------------------------------------------
# Build output path
# ---------------------------------------------------------------------------
mkdir -p results

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
log "Starting evaluation"
log "  Task:        $TASK"
log "  Model:       $MODEL"
log "  Base URL:    $BASE_URL"
log "  Controller:  $CONTROLLER"
log "  Parallelism: $JOBS"
log "  Output dir:  results/"
[[ -n "$RESUME_FILE" ]] && log "  Resuming:    $RESUME_FILE"
log ""

EXTRA_ARGS=()
[[ -n "$RESUME_FILE" ]] && EXTRA_ARGS+=(--output-file "$RESUME_FILE")

python vendor/AgentRL/examples/eval/server_agent.py \
    -m  "$MODEL" \
    -u  "$BASE_URL" \
    -j  "$JOBS" \
    -c  "$CONTROLLER" \
    -o  "results" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"} \
    "$TASK"

log ""
log "✅  Evaluation complete. Results saved under: results/"
log ""
log "To check stats, run:"
log "  bash scripts/check_results.sh results/<output-file>.jsonl"