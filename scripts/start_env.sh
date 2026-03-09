#!/usr/bin/env bash
# =============================================================================
# scripts/start_env.sh — Start the AgentRL controller + OS task workers
# Usage: bash scripts/start_env.sh [--os-only] [--down]
# =============================================================================
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[env]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }

COMPOSE_FILE="vendor/AgentBench/extra/docker-compose.yml"
OS_ONLY=false
BRING_DOWN=false

for arg in "$@"; do
    case $arg in
        --os-only) OS_ONLY=true ;;
        --down)    BRING_DOWN=true ;;
    esac
done

# Load .env if present
[[ -f .env ]] && export $(grep -v '^#' .env | xargs)

if $BRING_DOWN; then
    log "Bringing down all services..."
    docker compose -f "$COMPOSE_FILE" down
    exit 0
fi

if $OS_ONLY; then
    log "Starting OS-only stack (controller + os_interaction-std + redis)..."
    # Only bring up the services needed for OS tasks
    docker compose -f "$COMPOSE_FILE" up -d controller os_interaction-std redis
else
    log "Starting full stack..."
    warn "Note: webshop requires ~16GB RAM. Use --os-only if your machine is limited."
    docker compose -f "$COMPOSE_FILE" up -d
fi

log ""
log "Waiting for controller to be ready on :5020..."
for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:5020/api >/dev/null 2>&1; then
        log "✅  Controller is up at http://127.0.0.1:5020"
        break
    fi
    echo -n "."
    sleep 2
done

log ""
log "Running services:"
docker compose -f "$COMPOSE_FILE" ps

log ""
log "Controller dashboard: http://127.0.0.1:5020"
log "To stop:              bash scripts/start_env.sh --down"