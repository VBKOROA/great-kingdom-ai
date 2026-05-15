#!/bin/bash
set -euo pipefail

SOURCE=${1:-data/runpod/train-v2-gumbel-512k/checkpoints/training-latest.pt}
DEST_DIR=${2:-data/runpod/train-v2-gumbel-512k/checkpoints/snapshots}
INTERVAL_SECONDS=${3:-600}
KEEP_COUNT=${4:-0}
STABLE_CHECK_SECONDS=${SNAPSHOT_STABLE_CHECK_SECONDS:-2}
RETRY_SECONDS=${SNAPSHOT_RETRY_SECONDS:-30}
RETRY_SLEEP_SECONDS=${SNAPSHOT_RETRY_SLEEP_SECONDS:-1}

mkdir -p "$DEST_DIR"

echo "snapshot source: $SOURCE"
echo "snapshot dir:    $DEST_DIR"
echo "interval:        ${INTERVAL_SECONDS}s"
echo "stable check:    ${STABLE_CHECK_SECONDS}s"
echo "retry window:    ${RETRY_SECONDS}s"
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

copy_stable_source() {
    local destination=$1
    local temporary=$2
    local before
    local after
    local copied
    local deadline
    local status="missing"

    deadline=$((SECONDS + RETRY_SECONDS))
    while [ "$SECONDS" -le "$deadline" ]; do
        before=$(file_signature)
        if [ "$before" = "missing" ]; then
            status="missing"
            sleep "$RETRY_SLEEP_SECONDS"
            continue
        fi

        sleep "$STABLE_CHECK_SECONDS"
        after=$(file_signature)
        if [ "$before" != "$after" ]; then
            status="changing"
            sleep "$RETRY_SLEEP_SECONDS"
            continue
        fi

        cp "$SOURCE" "$temporary"
        copied=$(file_signature)
        if [ "$after" != "$copied" ]; then
            status="changed during copy"
            rm -f "$temporary"
            sleep "$RETRY_SLEEP_SECONDS"
            continue
        fi

        mv "$temporary" "$destination"
        return 0
    done

    rm -f "$temporary"
    echo "$status"
    return 1
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

    if failure_reason=$(copy_stable_source "$destination" "$temporary"); then
        echo "[$(date +%H:%M:%S)] saved $destination"
        prune_old_snapshots
    else
        echo "[$(date +%H:%M:%S)] skipped after retry: source $failure_reason"
    fi

    sleep "$INTERVAL_SECONDS"
done
