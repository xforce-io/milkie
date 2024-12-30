#!/bin/bash

port_bird=8234
port_expert=8123

# 检查端口是否可用
wait_for_port() {
    local port=$1
    local service_name=$2
    local max_attempts=30  # 最多等待30秒
    local attempt=1
    
    echo "Waiting for $service_name to be ready..."
    while ! nc -z localhost $port >/dev/null 2>&1; do
        if [ $attempt -ge $max_attempts ]; then
            echo "Timeout waiting for $service_name to start"
            return 1
        fi
        sleep 1
        attempt=$((attempt + 1))
    done
    echo "$service_name is ready"
    return 0
}

# 检查服务是否已在运行
check_services() {
    local bird_running=false
    local expert_running=false
    
    if nc -z localhost $port_bird 2>/dev/null; then
        echo "Bird server is already running on port $port_bird"
        bird_running=true
    fi
    
    if nc -z localhost $port_expert 2>/dev/null; then
        echo "SQL expert is already running on port $port_expert"
        expert_running=true
    fi
    
    if [ "$bird_running" = true ] || [ "$expert_running" = true ]; then
        echo "Please stop the running services first with: $0 stop"
        exit 1
    fi
}

start_services() {
    # 先检查服务是否已在运行
    check_services
    
    echo "Starting Bird server..."
    nohup python -m clients.bird.server &
    wait_for_port $port_bird "Bird server" || { echo "Failed to start Bird server"; exit 1; }
    
    echo "Starting SQL expert..."
    nohup ./examples/sql_expert/server_expert.sh cot_expert &
    wait_for_port $port_expert "SQL expert" || { echo "Failed to start SQL expert"; exit 1; }
    
    echo "All Bird services started successfully"
}

stop_services() {
    echo "Stopping Bird services..."
    # 结束 bird server 进程
    pkill -f "python.*clients.bird.server"
    # 结束 sql expert 进程
    pkill -f "milkie.*--agent.*cot_expert"
    
    # 等待端口释放
    while nc -z localhost $port_bird 2>/dev/null || nc -z localhost $port_expert 2>/dev/null; do
        sleep 1
    done
    
    echo "Bird services stopped"
}

case "$1" in
    "start")
        start_services
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        stop_services
        sleep 2  # 额外等待时间，确保进程完全清理
        start_services
        ;;
    *)
        echo "Usage: $0 start|stop|restart"
        exit 1
        ;;
esac
