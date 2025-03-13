#!/bin/bash

./bin/milkie \
    --folder examples/deep/ \
    --agent answer \
    --verbose \
    --root "$HOME" \
    --query "$1"

