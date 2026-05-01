#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUST_CRATE_DIR="$ROOT_DIR/rust/great_kingdom_core"

cd "$ROOT_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python command not found: $PYTHON_BIN" >&2
  exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "Rust cargo not found. Install rustup first, then rerun this script." >&2
  exit 1
fi

echo "Creating venv with system site packages: $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR" --system-site-packages
source "$VENV_DIR/bin/activate"

echo "Installing project dev dependencies without replacing Runpod's PyTorch"
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

echo "Checking PyTorch/CUDA visible from venv"
python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY

echo "Building Python extension with maturin"
cd "$RUST_CRATE_DIR"
python -m maturin develop

PY_LIBDIR="$(python - <<'PY'
import sysconfig

print(sysconfig.get_config_var("LIBDIR"))
PY
)"
PY_VERSION="$(python - <<'PY'
import sysconfig

print(sysconfig.get_config_var("LDVERSION") or sysconfig.get_config_var("VERSION"))
PY
)"

export PYO3_PYTHON="$VENV_DIR/bin/python"
export LD_LIBRARY_PATH="$PY_LIBDIR:${LD_LIBRARY_PATH:-}"
export RUSTFLAGS="-L native=$PY_LIBDIR -l python$PY_VERSION ${RUSTFLAGS:-}"

echo "Running Rust tests with explicit libpython link flags"
cargo test

cd "$ROOT_DIR"
echo "Running Python tests"
python -m pytest

echo "Runpod setup complete."
echo "For CUDA smoke test, run:"
echo "  source \"$VENV_DIR/bin/activate\" && python scripts/run_m6_smoke.py"
