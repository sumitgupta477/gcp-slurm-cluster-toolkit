#!/bin/bash
# One-time: create the Python training environment on the shared /home.
# Run on the LOGIN VM. Because /home is shared, every GPU node sees the same environment.
#   bash setup_env.sh
set -euo pipefail

VENV="${VENV:-$HOME/venv-train}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# Ubuntu needs the venv package before `python3 -m venv` works.
if ! python3 -m venv --help >/dev/null 2>&1 || ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  echo "Installing python3-venv (needs sudo, i.e. the OS Admin Login role)..."
  sudo apt-get update -y && sudo apt-get install -y python3-venv
fi

python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r "$HERE/requirements-train.txt"

"$VENV/bin/python" -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)"
echo "Done. Environment: $VENV"
echo "Check the GPU from a job:  srun --gpus=1 $VENV/bin/python -c 'import torch; print(torch.cuda.get_device_name(0))'"
