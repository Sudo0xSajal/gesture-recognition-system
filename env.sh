#!/usr/bin/env bash
# Source this file FIRST in every terminal before running any script.
# Usage: source env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR/code:$PYTHONPATH"
export LOG_ROOT="$SCRIPT_DIR/log"

mkdir -p "$LOG_ROOT"
echo "[env] PROJECT_ROOT = $PROJECT_ROOT"
echo "[env] PYTHONPATH   = $PYTHONPATH"
echo "[env] Ready."
