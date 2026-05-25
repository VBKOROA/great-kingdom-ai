#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-data/runpod/train-v6/replay}"
SIZE_GIB="${2:-2}"
BLOCK_MIB="${3:-64}"
COMPARE_DIR="${4:-/tmp/great-kingdom-ai/io-test}"

run_test() {
  local label="$1"
  local dir="$2"
  local size_gib="$3"
  local block_mib="$4"
  local block_bytes=$((block_mib * 1024 * 1024))
  local count=$((size_gib * 1024 / block_mib))
  local file="$dir/.gka-volume-io-test-$$.bin"

  mkdir -p "$dir"
  trap 'rm -f "$file"' RETURN

  echo
  echo "== $label =="
  echo "dir=$dir size=${size_gib}GiB bs=${block_mib}MiB count=$count"
  df -hT "$dir" || true
  findmnt -T "$dir" || true

  echo
  echo "-- write + fdatasync --"
  dd if=/dev/zero of="$file" bs="$block_bytes" count="$count" conv=fdatasync status=progress

  echo
  echo "-- read direct, fallback buffered if unsupported --"
  if ! dd if="$file" of=/dev/null bs="$block_bytes" iflag=direct status=progress; then
    dd if="$file" of=/dev/null bs="$block_bytes" status=progress
  fi

  echo
  echo "-- overwrite + fdatasync --"
  dd if=/dev/zero of="$file" bs="$block_bytes" count="$count" conv=fdatasync status=progress

  rm -f "$file"
  trap - RETURN
}

run_test "target volume" "$TARGET_DIR" "$SIZE_GIB" "$BLOCK_MIB"

if [[ -n "$COMPARE_DIR" ]]; then
  run_test "compare volume" "$COMPARE_DIR" "$SIZE_GIB" "$BLOCK_MIB"
fi
