#!/bin/bash
# Run all PPO benchmarks, each in an isolated Docker container.
# Usage: ./run_all_benchmarks.sh [--repeats N] [--run_from INDEX] ["optional notes for wandb"]
set -e

COMPOSE_FILE="setup/docker/docker-compose.run.yml"
NOTES=""
REPEATS=1
RUN_FROM=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --repeats)
            REPEATS="$2"
            shift 2
            ;;
        --run_from)
            RUN_FROM="$2"
            shift 2
            ;;
        *)
            NOTES="$1"
            shift
            ;;
    esac
done

TOTAL=$(docker compose -f "$COMPOSE_FILE" run --rm run python -m benchback_rl --n_index | tail -1)

for repeat in $(seq 1 $REPEATS); do
    echo "=== Repeat $repeat / $REPEATS ==="
    for i in $(seq $RUN_FROM $((TOTAL - 1))); do
        echo "=== Benchmark $i / $((TOTAL - 1)) ==="
        docker compose -f "$COMPOSE_FILE" run --rm run python -m benchback_rl --run_index "$i" --notes "$NOTES"
    done
    # Reset RUN_FROM after first repeat so subsequent repeats run all benchmarks
    RUN_FROM=0
done
