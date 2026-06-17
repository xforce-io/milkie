#!/usr/bin/env sh
set -eu

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not running; start Docker/Colima before running Redis e2e." >&2
  exit 1
fi

docker compose -f docker-compose.test.yml up -d redis

status=0
node tests/e2e/wait-for-redis.cjs &&
  REDIS_E2E_REQUIRED=1 jest tests/e2e/s-009-multi-turn-with-tool-error-recovery.e2e.test.ts --runInBand ||
  status=$?

docker compose -f docker-compose.test.yml down
exit "$status"
