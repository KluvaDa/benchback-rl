#!/bin/bash
# Export production requirements from the current dev environment
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_FILE="$ROOT_DIR/requirements.txt"

echo "Exporting installed packages..."

pip freeze 2>/dev/null | while read -r line; do
    [[ -z "$line" ]] && continue
    
    # Convert editable comments to pinned versions
    if [[ "$line" =~ ^#.*\(([a-zA-Z0-9_-]+)==([^\)]+)\)$ ]]; then
        echo "${BASH_REMATCH[1]}==${BASH_REMATCH[2]}"
        continue
    fi
    
    # Skip comments, editable installs, and local file references
    [[ "$line" =~ ^# || "$line" == -e* || "$line" =~ @ ]] && continue
    
    # Skip project itself
    [[ "$line" == benchback-rl* ]] && continue
    
    echo "$line"
done > "$OUTPUT_FILE"

echo "Exported $(wc -l < "$OUTPUT_FILE") packages to $OUTPUT_FILE"

