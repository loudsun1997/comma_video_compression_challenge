#!/usr/bin/env bash
# Back-compat: same as run_p10_smoke.sh (uses python3 in Colab when uv is missing).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/run_p10_smoke.sh" "$@"
