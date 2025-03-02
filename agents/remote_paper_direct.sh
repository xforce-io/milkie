#!/bin/bash

# 默认不启动 crawler
run_crawler=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --crawler)
            run_crawler=true
            shift
            ;;
    esac
done

# crawler（仅在指定 --crawler 参数时执行）
if [ "$run_crawler" = true ]; then
    cd ~/dev/github/crawlers
    bash bin/paper.sh
    zip -r paper.zip data/papers && scp paper.zip ari@10.4.119.109:~/dev/github/crawlers
    ssh ari@10.4.119.109 "cd ~/dev/github/crawlers/; unzip -o paper.zip"

    cd ~/dev/github/milkie
    find raw -type f ! -name "*.pdf" -delete

    python examples/paper_report/download.py `date +%Y-%m-%d` 9 /Users/xupeng
    zip -r raw.zip data/paper_report/raw/ && scp raw.zip ari@10.4.119.109:~/dev/github/milkie
    ssh ari@10.4.119.109 "cd ~/dev/github/milkie/; unzip -o raw.zip"
fi

ssh ari@10.4.119.109 "cd ~/dev/github/milkie; ./examples/bin/paper_report.sh"
scp ari@10.4.119.109:~/dev/github/milkie/data/paper_report/`date +%Y-%m-%d`/report.md ~/Downloads/report-`date +%Y-%m-%d`.md