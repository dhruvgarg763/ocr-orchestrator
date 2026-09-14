#!/usr/bin/env bash
# Sweep worker_lead_reserve and report both latency scenarios plus throughput.
#
# Why both scenarios: an earlier A/B of this knob measured only jobs arriving
# at t=0 into an EMPTY system, where every slot is free and a reservation has
# nothing to do. It concluded "no benefit", which was a property of the
# experiment rather than of the feature. The latecomer case - a job arriving
# into a system already saturated with VLM-stage work - is the one the
# reservation exists for, so it is measured here alongside the burst.
#
# Why throughput is reported: the reservation is carved OUT of
# worker_concurrency, so raising it caps what the main lane may hold. Little's
# law says saturating a 10 rps endpoint with ~2.25s calls needs ~23 concurrent
# slots, so a large reserve should start costing pages/sec. A latency win paid
# for with throughput is not a win.
set -u
export MSYS_NO_PATHCONV=1

BURST_JOBS=50      # 50-way lead-lane contention: what the reserve actually affects
BURST_PAGES=8      # smaller than the graded 20 to keep the sweep affordable
WINNER_PAGES=20    # the winner is re-validated at the graded shape

run_config() {
  local conc=$1 reserve=$2 label=$3

  ORCH_WORKER_CONCURRENCY=$conc ORCH_WORKER_LEAD_RESERVE=$reserve \
    docker compose up -d worker --scale worker=3 --force-recreate >/dev/null 2>&1
  sleep 7
  docker compose exec -T redis redis-cli FLUSHDB >/dev/null

  echo "=============================================================="
  echo "  $label   (concurrency=$conc, lead_reserve=$reserve)"
  echo "=============================================================="

  # Scenario the reservation targets: arriving into an already-busy system.
  docker compose exec -T api python /tmp/late.py busy 2>&1 \
    | grep -E "settled after|latecomer"

  docker compose exec -T redis redis-cli FLUSHDB >/dev/null
  sleep 3

  # Burst: 50 simultaneous jobs, plus the throughput check.
  docker compose exec -T api python /tmp/m.py "$BURST_JOBS" "$BURST_PAGES" "$BURST_JOBS" 2>&1 \
    | grep -E "^wall|ttfp from POST|ttfp from stream|events ==|seq gaps"

  docker compose exec -T redis redis-cli FLUSHDB >/dev/null
  sleep 2
  echo
}

docker compose cp scripts/measure_ttfp.py api:/tmp/m.py >/dev/null 2>&1
docker compose cp scripts/measure_latecomer.py api:/tmp/late.py >/dev/null 2>&1

run_config 16 0  "A. no reservation (shared budget, lane still read first)"
run_config 16 4  "B. current default"
run_config 16 8  "C. half the budget reserved"
run_config 16 12 "D. reserve above the Little's-law floor for main"
run_config 20 8  "E. reservation FUNDED by raising concurrency"
