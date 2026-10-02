#!/bin/bash
set -euo pipefail
printf '%s\n' '#!/bin/sh' 'exec node /opt/pi/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js "$@"' > /usr/local/bin/pi
chmod +x /usr/local/bin/pi
proxy_host=$(awk '/^nameserver / && $2 != "8.8.8.8" && $2 != "1.1.1.1" {print $2; exit}' /etc/resolv.conf)
export HTTP_PROXY="http://${proxy_host}:8899"
export HTTPS_PROXY="http://${proxy_host}:8899"
export NODE_USE_ENV_PROXY=1
version_home=$(mktemp -d)
mkdir -p "$version_home/grok" "$version_home/pi"
HOME="$version_home" GROK_HOME="$version_home/grok" grok --version
HOME="$version_home" PI_CODING_AGENT_DIR="$version_home/pi" pi --version
node /src/tests/e2e/agent-cli-storage-linux.cjs
