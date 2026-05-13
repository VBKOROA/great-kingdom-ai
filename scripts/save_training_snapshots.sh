#!/bin/bash
set -euo pipefail

SOURCE=${1:-data/runpod/train-v2-gumbel-512k/checkpoints/training-latest.pt}
DEST_DIR=${2:-data/runpod/train-v2-gumbel-512k/checkpoints/snapshots}
INTERVAL_SECONDS=${3:-600}
KEEP_COUNT=${4:-0}

mkdir -p "$DEST_DIR"

echo "snapshot source: $SOURCE"
echo "snapshot dir:    $DEST_DIR"
echo "interval:        ${INTERVAL_SECONDS}s"
if [ "$KEEP_COUNT" -gt 0 ]; then
    echo "keep count:      $KEEP_COUNT"
else
    echo "keep count:      unlimited"
fi

file_signature() {
    if [ ! -f "$SOURCE" ]; then
        echo "missing"
        return
    fi
    stat -c "%s:%Y" "$SOURCE"
}

wait_for_stable_source() {
    local before
    local after

    before=$(file_signature)
    if [ "$before" = "missing" ]; then
        return 1
    fi
    sleep 2
    after=$(file_signature)
    [ "$before" = "$after" ]
}

prune_old_snapshots() {
    if [ "$KEEP_COUNT" -le 0 ]; then
        return
    fi

    local count
    count=$(find "$DEST_DIR" -maxdepth 1 -type f -name 'training-latest-*.pt' | wc -l)
    if [ "$count" -le "$KEEP_COUNT" ]; then
        return
    fi

    find "$DEST_DIR" -maxdepth 1 -type f -name 'training-latest-*.pt' -printf '%T@ %p\n' \
        | sort -n \
        | head -n "$((count - KEEP_COUNT))" \
        | cut -d' ' -f2- \
        | xargs -r rm -f
}

while true; do
    timestamp=$(date +%Y%m%d-%H%M%S)
    destination="$DEST_DIR/training-latest-$timestamp.pt"
    temporary="$destination.tmp"

    if wait_for_stable_source; then
        cp "$SOURCE" "$temporary"
        mv "$temporary" "$destination"
        echo "[$(date +%H:%M:%S)] saved $destination"
        prune_old_snapshots
    else
        echo "[$(date +%H:%M:%S)] skipped: source missing or changing"
    fi

    sleep "$INTERVAL_SECONDS"
done
