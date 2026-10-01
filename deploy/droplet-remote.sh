#!/usr/bin/env bash
# Runs on the Droplet, started by droplet-redeploy.sh. Argument: git short revision.
# Builds the image, swaps the container, waits for health, then warms the first session
# so the first visitor does not pay the ~10 s cold import on a 1 vCPU box.
set -uo pipefail
REV="$1"
fail() { echo "FAILED: $*"; exit 1; }
cd /opt/llmplan && rm -rf src && mkdir src && tar -xzf /root/llmplan-src.tar.gz -C src || fail "unpack"
echo "building llmplan:$REV"
docker build -q -t "llmplan:$REV" -t llmplan:main src >/dev/null 2>>/root/deploy.log || fail "docker build"
docker rm -f llmplan >/dev/null 2>&1 || true
docker run -d --name llmplan --restart unless-stopped --memory 512m \
  -p 127.0.0.1:8501:8501 \
  -e LLMPLAN_USAGE_LOG=/var/lib/llmplan/usage.jsonl \
  -v /var/lib/llmplan:/var/lib/llmplan \
  "llmplan:$REV" >/dev/null || fail "docker run"
for i in $(seq 1 30); do
  curl -sf http://127.0.0.1:8501/_stcore/health >/dev/null && break
  sleep 2
done
curl -sf http://127.0.0.1:8501/_stcore/health >/dev/null || { docker logs --tail 50 llmplan; fail "container not healthy"; }
echo "healthy: llmplan:$REV"
# Warm-up: open one Streamlit session and hold it for 25 s so the server imports the app
# and its libraries once. The handshake alone is enough to start the first script run.
python3 - <<'PY' || echo "warm-up skipped"
import socket, base64, os, time
key = base64.b64encode(os.urandom(16)).decode()
req = ("GET /_stcore/stream HTTP/1.1\r\nHost: localhost\r\nConnection: Upgrade\r\nUpgrade: websocket\r\n"
       f"Sec-WebSocket-Version: 13\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Protocol: streamlit\r\n\r\n")
s = socket.create_connection(("127.0.0.1", 8501), timeout=5)
s.sendall(req.encode()); s.recv(300); time.sleep(25); s.close()
print("warm-up session held for 25 s")
PY
docker image prune -f >/dev/null 2>&1 || true
echo "DONE: llmplan:$REV"
