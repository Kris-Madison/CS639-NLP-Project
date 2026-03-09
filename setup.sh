#!/usr/bin/env bash
# =============================================================================
# setup.sh — One-shot environment setup for the NLP OS Agent project
# Usage: bash setup.sh
# =============================================================================
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC}  $*"; }
die()  { echo -e "${RED}[error]${NC} $*"; exit 1; }

# ---------------------------------------------------------------------------
# 1. Pull submodules (AgentBench + AgentRL)
# ---------------------------------------------------------------------------
log "Initialising git submodules..."
git submodule update --init --recursive

# ---------------------------------------------------------------------------
# 2. Verify system dependencies
# ---------------------------------------------------------------------------
log "Checking system dependencies..."

command -v docker  >/dev/null 2>&1 || die "Docker not found. Install from https://docs.docker.com/get-docker/"
command -v conda   >/dev/null 2>&1 || warn "conda not found — using system Python (3.9 recommended)"
command -v python3 >/dev/null 2>&1 || die "Python 3 not found."

DOCKER_COMPOSE_OK=false
docker compose version >/dev/null 2>&1 && DOCKER_COMPOSE_OK=true
$DOCKER_COMPOSE_OK || die "Docker Compose v2 not found. Update Docker Desktop or install the compose plugin."

log "Docker: $(docker --version)"
log "Docker Compose: $(docker compose version)"

# ---------------------------------------------------------------------------
# 3. Python environment
# ---------------------------------------------------------------------------
if command -v conda >/dev/null 2>&1; then
    if conda env list | grep -q "^agent-bench "; then
        warn "conda env 'agent-bench' already exists, skipping creation."
    else
        log "Creating conda env 'agent-bench' with Python 3.12..."
        conda create -n agent-bench python=3.12 -y
    fi
    log "Installing AgentRL eval script dependencies..."
    conda run -n agent-bench pip install pandas aiohttp openai tqdm
else
    warn "conda not available — installing into current Python env"
    pip install pandas aiohttp openai tqdm
fi

# ---------------------------------------------------------------------------
# 5. Build OS Docker images
# ---------------------------------------------------------------------------
log "Building OS interaction Docker images (this may take a few minutes)..."

DOCKERFILES_DIR="vendor/AgentBench/data/os_interaction/res/dockerfiles"

docker build -t local-os/default  -f "${DOCKERFILES_DIR}/default"  "${DOCKERFILES_DIR}"
docker build -t local-os/packages -f "${DOCKERFILES_DIR}/packages" "${DOCKERFILES_DIR}"
docker build -t local-os/ubuntu   -f "${DOCKERFILES_DIR}/ubuntu"   "${DOCKERFILES_DIR}"

log "Built images:"
docker images | grep "local-os"

# ---------------------------------------------------------------------------
# 6. Verify API key is set
# ---------------------------------------------------------------------------
if [[ -z "${OPENAI_API_KEY:-}" && -z "${ANTHROPIC_API_KEY:-}" ]]; then
    warn "Neither OPENAI_API_KEY nor ANTHROPIC_API_KEY is set."
    warn "Set one in your shell or copy .env.example to .env and fill it in."
fi

# ---------------------------------------------------------------------------
log ""
log "✅  Setup complete!"
log ""
log "Next steps:"
log "  1. Activate the conda env:        conda activate agent-bench"
log "  2. Copy .env.example → .env and add your API key"
log "  3. Start the environment stack:   bash scripts/start_env.sh"
log "  4. Run evaluation:                bash scripts/run_eval.sh [-m gpt-5-mini] [--task os-std] [-j 8]"
log ""
log "⚠️  Always run 'conda activate agent-bench' before using any scripts."
log ""