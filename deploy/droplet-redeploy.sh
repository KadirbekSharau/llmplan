#!/usr/bin/env bash
# Redeploy the current HEAD of this checkout to the llmplan Droplet.
# Usage: deploy/droplet-redeploy.sh [host]   (default host: 137.184.154.129, user root)
# Requires: ssh access with the key registered on the Droplet; docker, caddy and the
# /var/lib/llmplan volume already set up per docs/DEPLOY.md "DigitalOcean Droplet".
#
# The remote steps run under setsid with a log file, so a dropped SSH session does not
# abort the build or leave the old container running. The script then polls the log.
set -euo pipefail
HOST="${1:-137.184.154.129}"
REV="$(git rev-parse --short HEAD)"
TMP="$(mktemp -t llmplan-src.XXXXXX).tar.gz"
git archive --format=tar.gz -o "$TMP" HEAD
scp -q "$TMP" "root@$HOST:/root/llmplan-src.tar.gz"
rm -f "$TMP"
scp -q "$(dirname "$0")/droplet-remote.sh" "root@$HOST:/opt/llmplan/remote.sh"
ssh "root@$HOST" "chmod +x /opt/llmplan/remote.sh && rm -f /root/deploy.log && setsid nohup /opt/llmplan/remote.sh $REV > /root/deploy.log 2>&1 < /dev/null & disown; echo started"
# Poll the remote log until it reports DONE or FAILED.
for _ in $(seq 1 120); do
  status="$(ssh "root@$HOST" "grep -o -E '^(DONE|FAILED).*' /root/deploy.log 2>/dev/null | tail -1 || true")"
  if [ -n "$status" ]; then
    ssh "root@$HOST" "tail -20 /root/deploy.log"
    [[ "$status" == DONE* ]] && exit 0 || exit 1
  fi
  sleep 10
done
echo "timed out waiting for the remote deploy; see /root/deploy.log on $HOST" >&2
exit 1
