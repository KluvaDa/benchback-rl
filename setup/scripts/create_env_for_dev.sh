#!/bin/bash
# Create setup/docker/.env file with user/group IDs for Docker
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$SETUP_DIR/docker/.env"

# Get user and group IDs
USER_UID=$(id -u)
USER_GID=$(id -g)
DOCKER_GID=$(stat -c '%g' /var/run/docker.sock 2>/dev/null || echo '999')

# Check if .env already exists
if [[ -f "$ENV_FILE" ]]; then
    echo "Updating existing $ENV_FILE..."
    # Remove old UID/GID/DOCKER_GID lines if they exist
    grep -v '^UID=' "$ENV_FILE" | grep -v '^GID=' | grep -v '^DOCKER_GID=' > "$ENV_FILE.tmp" || true
    mv "$ENV_FILE.tmp" "$ENV_FILE"
else
    echo "Creating $ENV_FILE..."
fi

# Append the user/group settings
cat >> "$ENV_FILE" << EOF
UID=$USER_UID
GID=$USER_GID
DOCKER_GID=$DOCKER_GID
EOF

echo "Created $ENV_FILE with:"
echo "  UID=$USER_UID"
echo "  GID=$USER_GID"
echo "  DOCKER_GID=$DOCKER_GID"
