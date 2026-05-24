#!/usr/bin/env bash
#
# Render a recorded run as a self-contained HTML report.
#
#   $ ./examples/s-002-inspect/report.sh [runId]
#
# Defaults to the runId in .milkie/last-run.txt (written by record.ts).
# Pass an explicit runId as $1 to render a different recorded run.
# Writes the HTML to ./report.html in this directory.
#
# Story: docs/stories/s-002-inspect-a-completed-run.md

set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="${1:-$(cat "$EXAMPLE_DIR/.milkie/last-run.txt")}"

# Run from the example dir so the CLI's `.milkie/` upward-search lands here.
cd "$EXAMPLE_DIR"

# Use the built CLI from the repo root.
REPO_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"
node "$REPO_ROOT/dist/cli/index.js" trace report "$RUN_ID" > report.html
echo "wrote report.html for run $RUN_ID"
