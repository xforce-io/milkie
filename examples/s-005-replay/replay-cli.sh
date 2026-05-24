#!/usr/bin/env bash
#
# Replay a recorded run via the milkie CLI.
#
#   $ ./examples/s-005-replay/replay-cli.sh [runId]
#
# Defaults to the runId in .milkie/last-run.txt (written by record.ts).
# Pass an explicit runId as $1 to replay a different recorded run.
#
# Story: docs/stories/s-005-deterministic-replay.md

set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="${1:-$(cat "$EXAMPLE_DIR/.milkie/last-run.txt")}"

# Run from the example dir so the CLI's `.milkie/` upward-search lands here.
cd "$EXAMPLE_DIR"

# Use the built CLI from the repo root.
REPO_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"
node "$REPO_ROOT/dist/cli/index.js" trace replay "$RUN_ID"
