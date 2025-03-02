#!/bin/bash

# 默认不启动 crawler
run_crawler=false
days=7

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --crawler)
            run_crawler=true
            shift
            ;;
        --days)
            days=$2
            shift
            shift
            ;;
    esac
done

# crawler（仅在指定 --crawler 参数时执行）
if [ "$run_crawler" = true ]; then
    cd ~/dev/github/crawlers
    bash bin/paper.sh
fi

cd ~/dev/github/milkie
find data/paper_report/raw -type f ! -name "*.pdf" -delete
python examples/paper_report/download.py `date +%Y-%m-%d` $days /Users/xupeng
./examples/bin/paper_report.sh --days $days