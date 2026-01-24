#!/bin/bash
# Run all PPO benchmarks, each in an isolated Docker container.
# Usage: ./run_all_benchmarks.sh [--repeats N] ["optional notes for wandb"]
set -e

COMPOSE_FILE="setup/docker/docker-compose.run.yml"
NOTES=""
REPEATS=""
REPEATS_ARG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --repeats)
            REPEATS="$2"
            REPEATS_ARG="--repeats $2"
            shift 2
            ;;
        *)
            NOTES="$1"
            shift
            ;;
    esac
done

TOTAL=$(docker compose -f "$COMPOSE_FILE" run --rm run python -m benchback_rl --n_index $REPEATS_ARG | tail -1)

for i in $(seq 0 $((TOTAL - 1))); do
    echo "=== Benchmark $i / $((TOTAL - 1)) ==="
    docker compose -f "$COMPOSE_FILE" run --rm run python -m benchback_rl --run_index "$i" $REPEATS_ARG --notes "$NOTES"
done
