#!/bin/bash
# Show the values needed for docker/.env

echo "Add these to docker/.env:"
echo ""
echo "UID=$(id -u)"
echo "GID=$(id -g)"
echo "DOCKER_GID=$(stat -c '%g' /var/run/docker.sock 2>/dev/null || echo '999')"
