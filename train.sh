#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
PYTHON="${PYTHON:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$("$PYTHON" -c 'import torch; print(max(torch.cuda.device_count(), 1))')}"
if (( NPROC_PER_NODE > 1 )); then
  exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" train.py "$@"
fi
exec "$PYTHON" train.py "$@"
