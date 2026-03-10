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
OS_ONLY=true
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

# ---------------------------------------------------------------------------
# Fix: pin aiodocker==0.21.0 inside the OS task worker container
#
# aiodocker 0.22.0+ changed timeout handling to require aiohttp.ClientTimeout
# objects, but agentrl-worker passes plain ints, causing:
#   AttributeError: 'int' object has no attribute 'connect'
# This patch is applied every time the container starts since containers
# are ephemeral and don't persist pip installs across restarts.
# ---------------------------------------------------------------------------
log ""
log "Applying aiodocker fix in OS task worker..."
# Wait for the container to be running before exec-ing into it
for i in $(seq 1 15); do
    WORKER_CONTAINER=$(docker ps --filter "ancestor=agentbench-fc-os_interaction-std" --format "{{.Names}}" | head -1)
    # Also try filtering by image name pattern in case naming differs
    [[ -z "$WORKER_CONTAINER" ]] && \
        WORKER_CONTAINER=$(docker ps --format "{{.Names}}" | grep -i "os_interaction\|os-interaction" | head -1)
    [[ -n "$WORKER_CONTAINER" ]] && break
    echo -n "."
    sleep 1
done

if [[ -n "$WORKER_CONTAINER" ]]; then
    docker exec "$WORKER_CONTAINER" pip install -q "aiodocker==0.21.0"
    log "✅  aiodocker pinned to 0.21.0 in $WORKER_CONTAINER"
else
    warn "Could not find OS task worker container to patch — aiodocker fix not applied."
    warn "If tasks fail, run manually: docker exec <container> pip install 'aiodocker==0.21.0'"
fi

log ""
log "Waiting for controller to be ready on :5020..."
READY=false
for i in $(seq 1 10); do
    if curl -s http://localhost:5020/api >/dev/null 2>&1; then
        READY=true
        break
    fi
    echo -n "."
    sleep 2
done
echo ""
if ! $READY; then
    warn "Controller did not respond. Check logs with:"
    warn "  docker compose -f $COMPOSE_FILE logs controller"
else
    log "✅  Controller is up at http://localhost:5020"
fi

log ""
log "Running services:"
docker compose -f "$COMPOSE_FILE" ps

log ""
log "To stop:              bash scripts/start_env.sh --down"
