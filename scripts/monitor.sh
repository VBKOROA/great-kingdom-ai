#!/bin/bash
INTERVAL=${1:-10}

read_cpu_stat() {
    read -r _cpu user nice system idle iowait irq softirq steal guest guest_nice _rest < /proc/stat
    idle_all=$((idle + iowait))
    total=$((user + nice + system + idle + iowait + irq + softirq + steal + guest + guest_nice))
    echo "$idle_all $total"
}

visible_vcpus() {
    nproc 2>/dev/null || grep -c '^processor' /proc/cpuinfo
}

quota_vcpus() {
    if [ -r /sys/fs/cgroup/cpu.max ]; then
        read -r quota period < /sys/fs/cgroup/cpu.max
        if [ "$quota" != "max" ] && [ "${period:-0}" -gt 0 ] 2>/dev/null; then
            awk -v quota="$quota" -v period="$period" 'BEGIN { printf "%.2f", quota / period }'
            return
        fi
    fi

    if [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ] && [ -r /sys/fs/cgroup/cpu/cpu.cfs_period_us ]; then
        quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
        period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
        if [ "$quota" -gt 0 ] && [ "$period" -gt 0 ] 2>/dev/null; then
            awk -v quota="$quota" -v period="$period" 'BEGIN { printf "%.2f", quota / period }'
            return
        fi
    fi

    visible_vcpus
}

clear
read -r prev_idle prev_total < <(read_cpu_stat)

while true; do
    sleep "$INTERVAL"
    read -r idle total < <(read_cpu_stat)

    idle_delta=$((idle - prev_idle))
    total_delta=$((total - prev_total))
    prev_idle=$idle
    prev_total=$total

    visible=$(visible_vcpus)
    quota=$(quota_vcpus)
    cpu_line=$(awk \
        -v idle_delta="$idle_delta" \
        -v total_delta="$total_delta" \
        -v visible="$visible" \
        -v quota="$quota" \
        'BEGIN {
            if (total_delta <= 0) {
                total_pct = 0.0
            } else {
                total_pct = 100.0 * (total_delta - idle_delta) / total_delta
            }
            busy_vcpus = total_pct * visible / 100.0
            quota_pct = quota > 0 ? 100.0 * busy_vcpus / quota : total_pct
            printf "CPU 사용량: %.1f%% total, %.2f/%s vCPU busy, %.1f%% of quota", total_pct, busy_vcpus, quota, quota_pct
        }')

    echo "=== 시스템 모니터링 ($(date +%H:%M:%S)) ==="
    echo "$cpu_line"

    # RAM 사용량
    ram_info=$(free -m | awk '/Mem:/ {print $3 "/" $2 " MB"}')
    echo "RAM 사용량: $ram_info"

    # GPU 정보
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader | \
        awk -F', ' '{print "VRAM 사용량: "$1" / "$2" | GPU 사용률: "$3}'
    else
        echo "GPU 정보: nvidia-smi 없음"
    fi

    echo
done
