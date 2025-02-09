#!/bin/bash

./bin/milkie \
    --folder examples/paper_report/ \
    --agent paper_report \
    --verbose \
    --date `date +%Y-%m-%d` \
    --days 12 \
    --root "$HOME"
