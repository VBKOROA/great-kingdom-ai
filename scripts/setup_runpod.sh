#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUST_CRATE_DIR="$ROOT_DIR/rust/great_kingdom_core"
RUNPOD_RUST_FEATURES="${RUNPOD_RUST_FEATURES:-extension-module,onnx-cuda}"
CARGO_TEST_FEATURES="${CARGO_TEST_FEATURES:-onnx-cuda}"

cd "$ROOT_DIR"

# 자동 설치 함수
install_python() {
  echo "Python을 설치해야 합니다."
  if command -v apt-get >/dev/null 2>&1; then
    echo "apt를 사용하여 Python 설치 중..."
    sudo apt-get update
    sudo apt-get install -y python3 python3-venv python3-dev
  elif command -v brew >/dev/null 2>&1; then
    echo "brew를 사용하여 Python 설치 중..."
    brew install python3
  else
    echo "자동 설치할 수 없습니다. Python3를 수동으로 설치하세요." >&2
    exit 1
  fi
}

install_rust() {
  echo "Rust/Cargo를 설치 중입니다..."
  if ! command -v rustup >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
  fi
  rustup update
  echo "Rust 설치 완료"
}

install_rust_native_deps() {
  local missing=()
  local packages=(libssl-dev openssl pkg-config)

  if ! command -v dpkg-query >/dev/null 2>&1; then
    echo "dpkg-query를 찾을 수 없어 native dependency 확인을 건너뜁니다."
    return
  fi

  for package in "${packages[@]}"; do
    if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q "install ok installed"; then
      missing+=("$package")
    fi
  done

  if ((${#missing[@]} == 0)); then
    echo "Rust native dependencies already installed: ${packages[*]}"
    return
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    echo "필요한 패키지를 찾지 못했지만 apt-get이 없습니다: ${missing[*]}" >&2
    exit 1
  fi

  echo "Installing Rust native dependencies: ${missing[*]}"
  sudo apt-get update
  sudo apt-get install -y "${missing[@]}"
}

# Python 확인 및 설치
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 명령을 찾을 수 없습니다: $PYTHON_BIN"
  install_python
fi

# Cargo 확인 및 설치
if ! command -v cargo >/dev/null 2>&1; then
  install_rust
fi

install_rust_native_deps

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

echo "Building Python extension with maturin features: $RUNPOD_RUST_FEATURES"
cd "$RUST_CRATE_DIR"
python -m maturin develop --features "$RUNPOD_RUST_FEATURES"

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

echo "Running Rust tests with explicit libpython link flags and features: $CARGO_TEST_FEATURES"
cargo test --features "$CARGO_TEST_FEATURES"

cd "$ROOT_DIR"
echo "Running Python tests"
python -m pytest

echo "Runpod setup complete."
echo "For CUDA smoke test, run:"
echo "  source \"$VENV_DIR/bin/activate\" && python scripts/run_m6_smoke.py --device cuda"
echo "For Rust ONNX pipeline training, run:"
echo "  great-kingdom-rust-onnx-pipeline --device cuda --pipeline-config configs/runpod/pipeline-runpod.json --train-config configs/runpod/train-runpod.json --arena-config configs/runpod/arena-runpod.json"
