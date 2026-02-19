#!/bin/bash
# Pre-generate trajectory data for training
#
# Usage:
#   bash scripts/generate_data.sh                       # 100k train (default)
#   bash scripts/generate_data.sh --n_train 10000       # smaller for testing
#   bash scripts/generate_data.sh --help                # see all options

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$PROJECT_DIR"

# Use .venv if it exists, otherwise system python
if [ -f "$PROJECT_DIR/.venv/bin/python" ]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
else
    PYTHON="python"
fi

echo "Using python: $PYTHON"
echo "Project dir:  $PROJECT_DIR"
echo ""

exec "$PYTHON" "$SCRIPT_DIR/generate_data.py" "$@"
