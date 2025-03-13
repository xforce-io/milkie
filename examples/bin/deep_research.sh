#!/bin/bash

./bin/milkie \
    --folder examples/deep/ \
    --agent researcher \
    --verbose \
    --root "$HOME" \
    --query "$1"

