#!/bin/bash

# 初始化 days 变量为空
days=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --days)
            days=$2
            shift
            shift
            ;;
        *)  # 处理未知参数
            shift
            ;;
    esac
done

# 检查 days 参数是否提供
if [ -z "$days" ]; then
    echo "错误：必须提供 --days 参数"
    echo "用法：$0 --days <天数>"
    exit 1
fi

./bin/milkie \
    --folder examples/paper_report/ \
    --agent paper_report \
    --verbose \
    --date `date +%Y-%m-%d` \
    --days $days \
    --root "$HOME"
