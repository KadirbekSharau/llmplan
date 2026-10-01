#!/usr/bin/env bash
# Redeploy the current HEAD of this checkout to the llmplan Droplet.
# Usage: deploy/droplet-redeploy.sh [host]   (default host: 137.184.154.129, user root)
# Requires: ssh access with the key registered on the Droplet; docker, caddy and the
# /var/lib/llmplan volume already set up per docs/DEPLOY.md "DigitalOcean Droplet".
set -euo pipefail
HOST="${1:-137.184.154.129}"
REV="$(git rev-parse --short HEAD)"
TMP="$(mktemp -t llmplan-src.XXXXXX).tar.gz"
git archive --format=tar.gz -o "$TMP" HEAD
scp -q "$TMP" "root@$HOST:/root/llmplan-src.tar.gz"
rm -f "$TMP"
ssh "root@$HOST" bash -s "$REV" <<'REMOTE'
set -euo pipefail
REV="$1"
cd /opt/llmplan && rm -rf src && mkdir src && tar -xzf /root/llmplan-src.tar.gz -C src
docker build -q -t "llmplan:$REV" -t llmplan:main src >/dev/null
docker rm -f llmplan >/dev/null 2>&1 || true
docker run -d --name llmplan --restart unless-stopped --memory 512m \
  -p 127.0.0.1:8501:8501 \
  -e LLMPLAN_USAGE_LOG=/var/lib/llmplan/usage.jsonl \
  -v /var/lib/llmplan:/var/lib/llmplan \
  llmplan:main >/dev/null
for i in $(seq 1 30); do
  curl -sf http://127.0.0.1:8501/_stcore/health >/dev/null && { echo "healthy: llmplan:$REV"; exit 0; }
  sleep 2
done
echo "container did not become healthy; docker logs llmplan:" >&2; docker logs --tail 50 llmplan >&2; exit 1
REMOTE
docker image prune -f >/dev/null 2>&1 || true
