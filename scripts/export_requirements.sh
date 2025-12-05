#!/bin/bash
# Export production requirements from the current dev environment
# Dynamically excludes dev-only dependencies by comparing resolution with/without [dev] extras
# Converts editable installs to pinned versions
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_FILE="$ROOT_DIR/requirements.txt"

echo "Determining dev-only packages dynamically..."

# Get packages that would be installed for production (no extras)
prod_packages=$(uv pip compile "$ROOT_DIR/pyproject.toml" --quiet 2>/dev/null | grep -v '^#' | grep -v '^ ' | grep -v '^-e' | grep '==' | cut -d'=' -f1 | tr '[:upper:]' '[:lower:]' | tr '_' '-' | sort -u)

# Get packages that would be installed with dev extras
dev_packages=$(uv pip compile "$ROOT_DIR/pyproject.toml" --extra dev --quiet 2>/dev/null | grep -v '^#' | grep -v '^ ' | grep -v '^-e' | grep '==' | cut -d'=' -f1 | tr '[:upper:]' '[:lower:]' | tr '_' '-' | sort -u)

# Dev-only packages are those in dev but not in prod
dev_only_packages=$(comm -23 <(echo "$dev_packages") <(echo "$prod_packages"))

echo "Dev-only packages to exclude:"
echo "$dev_only_packages" | sed 's/^/  - /'
echo ""

echo "Exporting requirements from current environment..."

# Freeze all packages, then filter and transform
pip freeze 2>/dev/null | while read -r line; do
    # Skip empty lines
    [[ -z "$line" ]] && continue
    
    # Handle editable install comments like "# Editable Git install with no remote (flax==0.11.2)"
    if [[ "$line" =~ ^#.*\(([a-zA-Z0-9_-]+)==([^\)]+)\)$ ]]; then
        pkg_name="${BASH_REMATCH[1]}"
        pkg_version="${BASH_REMATCH[2]}"
        pkg_name_lower=$(echo "$pkg_name" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
        
        # Check if it's a dev-only package
        if echo "$dev_only_packages" | grep -qx "$pkg_name_lower"; then
            echo "  Excluding dev package (editable): $pkg_name==$pkg_version" >&2
            continue
        fi
        
        # Skip the project itself
        if [[ "$pkg_name_lower" == "benchback-rl" ]]; then
            echo "  Skipping project itself: $pkg_name==$pkg_version" >&2
            continue
        fi
        
        echo "  Converting editable to pinned: $pkg_name==$pkg_version" >&2
        echo "$pkg_name==$pkg_version"
        continue
    fi
    
    # Skip other comments
    [[ "$line" =~ ^# ]] && continue
    
    # Skip editable installs (these should have been handled by comments above)
    if [[ "$line" == -e* ]]; then
        echo "  Skipping editable line: $line" >&2
        continue
    fi
    
    # Skip file:// references (local wheel installs)
    if [[ "$line" =~ @ ]]; then
        echo "  Skipping local install: $line" >&2
        continue
    fi
    
    # Extract package name (before ==)
    pkg_name=$(echo "$line" | cut -d'=' -f1 | tr '[:upper:]' '[:lower:]' | tr '_' '-')
    
    # Check if it's a dev-only package
    if echo "$dev_only_packages" | grep -qx "$pkg_name"; then
        echo "  Excluding dev package: $line" >&2
        continue
    fi
    
    echo "$line"
done > "$OUTPUT_FILE"

echo ""
echo "Requirements exported to: $OUTPUT_FILE"
echo "Total packages: $(wc -l < "$OUTPUT_FILE")"
