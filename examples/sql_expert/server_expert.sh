#!/bin/bash

# 检查参数数量
if [ $# -ne 1 ]; then
    echo "Usage: $0 <agent_name>"
    exit 1
fi

./bin/milkie \
    --folder examples/sql_expert \
    --server \
    --port 8123 \
    --agent "$1"
