#!/usr/bin/env bash
# =============================================================================
# scripts/run_eval.sh — Run AgentBench OS evaluation via AgentRL
#
# Usage:
#   bash scripts/run_eval.sh                          # defaults: gpt-4o-mini, os-std
#   bash scripts/run_eval.sh -m claude-sonnet-4-... \
#                            -t os-std -j 8
#   bash scripts/run_eval.sh --resume results/os-std-run1.jsonl
#
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
MODEL="${MODEL:-gpt-4o-mini}"
BASE_URL="${BASE_URL:-https://api.openai.com/v1}"
TASK="${TASK:-os-std}"
JOBS="${JOBS:-8}"          # concurrent sessions
CONTROLLER="${CONTROLLER:-http://localhost:5020/api}"
RESUME_FILE=""

# Load .env if present
[[ -f .env ]] && export $(grep -v '^#' .env | xargs)

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)    MODEL="$2";      shift 2 ;;
        -u|--url)      BASE_URL="$2";   shift 2 ;;
        -t|--task)     TASK="$2";       shift 2 ;;
        -j|--jobs)     JOBS="$2";       shift 2 ;;
        -c|--controller) CONTROLLER="$2"; shift 2 ;;
        --resume)      RESUME_FILE="$2"; shift 2 ;;
        *) die "Unknown argument: $1" ;;
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
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAFE_MODEL=$(echo "$MODEL" | tr '/:' '--')
OUTFILE="results/${TASK}-${SAFE_MODEL}-${TIMESTAMP}.jsonl"
mkdir -p results

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
log "Starting evaluation"
log "  Task:        $TASK"
log "  Model:       $MODEL"
log "  Base URL:    $BASE_URL"
log "  Parallelism: $JOBS"
log "  Output:      $OUTFILE"
[[ -n "$RESUME_FILE" ]] && log "  Resuming:    $RESUME_FILE"
log ""

EXTRA_ARGS=()
[[ -n "$RESUME_FILE" ]] && EXTRA_ARGS+=(--file "$RESUME_FILE")

python vendor/AgentRL/examples/eval/server_agent.py \
    -m  "$MODEL" \
    -u  "$BASE_URL" \
    -j  "$JOBS" \
    -c  "$CONTROLLER" \
    -o  "$OUTFILE" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    "$TASK"

log ""
log "✅  Evaluation complete. Results saved to: $OUTFILE"
log ""
log "To check stats, run:"
log "  bash scripts/check_results.sh $OUTFILE"