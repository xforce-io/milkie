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
    bash bin/caijing.sh

    zip -r news.zip data/news_articles && scp news.zip jp:~/
    ssh jp 'scp news.zip ari@10.4.119.109:~/dev/github/crawlers'
    ssh jp 'ssh ari@10.4.119.109 "cd ~/dev/github/crawlers/; unzip -o news.zip"'
fi

ssh jp 'ssh ari@10.4.119.109 "cd ~/dev/github/milkie; ./examples/bin/viewpoint.sh"'
