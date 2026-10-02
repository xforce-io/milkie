#!/bin/bash
# Run the #265 Linux container probe. Mounts login files only, not the host CLI home.
set -euo pipefail
cd "$(dirname "$0")/../.."
test -f "$HOME/.grok/auth.json"
test -f "$HOME/.pi/agent/auth.json"
test -f "$HOME/.pi/agent/settings.json"
stage="$PWD/test-output/verify-milkie/265-linux"
mkdir -p "$stage"
if [ ! -x "$stage/grok" ]; then
  version=$(curl -fsSL https://x.ai/cli/stable | tr -d '[:space:]')
  curl -fL -o "$stage/grok" "https://x.ai/cli/grok-${version}-linux-aarch64"
  chmod +x "$stage/grok"
fi
if [ ! -d "$stage/pi-prefix/node_modules/@earendil-works/pi-coding-agent" ]; then
  npm install --prefix "$stage/pi-prefix" @earendil-works/pi-coding-agent@0.85.1 --omit=dev
fi
docker run --rm --platform linux/arm64 \
  -v "$PWD":/src:ro \
  -v "$stage/grok":/usr/local/bin/grok:ro \
  -v "$stage/pi-prefix":/opt/pi:ro \
  -v "$HOME/.grok/auth.json":/auth/grok-auth.json:ro \
  -v "$HOME/.pi/agent/auth.json":/auth/pi-auth.json:ro \
  -v "$HOME/.pi/agent/settings.json":/auth/pi-settings.json:ro \
  -e MILKIE_LINUX_STORAGE=1 \
  -e MILKIE_LINUX_RUNTIMES="${MILKIE_LINUX_RUNTIMES:-}" \
  -w /src \
  node:22-bookworm \
  bash /src/tests/e2e/linux-container-entry.sh
