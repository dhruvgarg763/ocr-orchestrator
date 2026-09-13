#!/bin/bash
cd /e/sarvam-vision-orchestrator
# Endpoint is passed WITHOUT a leading slash and reassembled in Python.
# Git Bash rewrites a leading-slash argv as a Windows path: "/jobs/stream"
# arrived as "C:/Program Files/Git/jobs/stream" and every request 404'd.
ENDPOINT=$1; LABEL=$2
docker compose exec -T redis redis-cli FLUSHDB >/dev/null
docker compose restart api worker >/dev/null 2>&1; sleep 12
curl -s -X POST localhost:8001/admin/reset >/dev/null
rm -f bench/rss_$LABEL.log
( for i in $(seq 1 120); do
    docker stats --no-stream --format "{{.Name}}|{{.MemUsage}}" 2>/dev/null | grep -E "api-1|worker-|redis"
    sleep 2
  done ) > bench/rss_$LABEL.log 2>&1 &
MON=$!
/e/sarvam-vision-orchestrator/.venv/Scripts/python.exe - "$ENDPOINT" <<'PY'
import asyncio, httpx, collections, sys, time
ENDPOINT = "/" + sys.argv[1]
async def main():
    codes = collections.Counter()
    data = open("bench/job20.pdf","rb").read() if ENDPOINT.endswith("stream") else None
    async with httpx.AsyncClient(base_url="http://localhost:8000", timeout=180) as c:
        async def up(i):
            if ENDPOINT.endswith("stream"):
                r = await c.post(ENDPOINT, content=data,
                                 headers={"Content-Type": "application/pdf"})
            else:
                with open("bench/job20.pdf","rb") as f:
                    r = await c.post(ENDPOINT, files={"file": ("j.pdf", f, "application/pdf")})
            codes[r.status_code] += 1
        t0 = time.monotonic()
        await asyncio.gather(*(up(i) for i in range(50)))
        print(f"    50 uploads (1000 MiB) in {time.monotonic()-t0:.1f}s -> {dict(codes)}")
asyncio.run(main())
PY
for i in $(seq 1 30); do
  sleep 6
  D=$(docker compose exec -T redis redis-cli XLEN stream:pages | tr -d '\r')
  [ "$D" = "0" ] && break
done
kill $MON 2>/dev/null; sleep 1
