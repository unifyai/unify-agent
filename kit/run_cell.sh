#!/usr/bin/env bash
# Run one cleanslate-v1 office cell on a bench worker. OPS runs this; see kit/README.md.
#
#   kit/run_cell.sh --arm full|no-memory --run-index N --tasks all --order frozen \
#       (--fake | --confirm-paid --prereg PATH --prereg-sha256 HEX) [--workbench DIR] [--max-fs-mb N]
#
# Sets up (once) a virtual environment from the system interpreter. The harness needs only the
# standard library, so nothing is installed into it. Then it runs kit/run_cell.py in isolated mode
# (-I: no PYTHONPATH, no user site-packages). For paid cells, OPENROUTER_API_KEY must already be in the
# environment through OPS's usual secret route. This script never echoes it.
set -euo pipefail
umask 077
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${CLEANSLATE_VENV:-$HOME/.local/share/continual-harness-research/cleanslate-venv}"
SYSTEM_PYTHON=/usr/bin/python3
command -v bwrap >/dev/null || { echo '{"refused": "bubblewrap (bwrap) is not installed"}'; exit 2; }
if [ ! -x "$VENV/bin/python" ]; then
    "$SYSTEM_PYTHON" -m venv "$VENV"
fi
"$VENV/bin/python" -I -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
exec "$VENV/bin/python" -I -B "$KIT/run_cell.py" "$@"
