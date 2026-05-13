#!/bin/bash
INTERVAL=${1:-10}

clear
while true; do
    echo "=== 시스템 모니터링 ($(date +%H:%M:%S)) ==="
    
    # CPU 사용량
    cpu_load=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
    echo "CPU 사용량: $cpu_load% / 100%"
    
    # RAM 사용량
    ram_info=$(free -m | awk '/Mem:/ {print $3 "/" $2 " MB"}')
    echo "RAM 사용량: $ram_info"
    
    # GPU 정보
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader | \
    awk -F', ' '{print "VRAM 사용량: "$1" / "$2" | GPU 사용률: "$3}'
    
    sleep $INTERVAL
done