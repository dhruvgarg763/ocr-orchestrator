#!/usr/bin/env bash
# The headline crash-recovery demo: SIGKILL a worker mid-flight, prove the job
# still finishes and no committed stage was ever computed twice.
#
# TWO scenarios, because they exercise DIFFERENT recovery code and only the
# second one is Step 14:
#
#   restart      `docker kill -9` then `docker start` reuses the container, so
#                the new process has the SAME hostname - and therefore the same
#                consumer name. `read_own_pending` finds its own Pending Entries
#                List and resumes immediately. This is the fast path, and it is
#                what a bare `docker kill` demo actually tests.
#
#   replacement  `docker rm` then `docker compose up` builds a NEW container, so
#                the hostname and consumer name change. The dead consumer's PEL
#                now belongs to a name that will never return: the entries are
#                skipped by `XREADGROUP >` because they were already delivered,
#                and no live consumer will ever ack them. Only the reaper can
#                recover these, and a demo that omits this scenario would report
#                success while Step 14 was entirely unexercised.
#
# Recovery in the replacement case takes up to reaper_min_idle_s +
# reaper_interval_s (45s on the defaults), so both are shortened here. The
# shortening is the ONLY thing tuned: the mechanism, the thresholds' meaning and
# every assertion are the shipped ones.
set -u
export MSYS_NO_PATHCONV=1

JOBS=${JOBS:-3}
PAGES=${PAGES:-20}
DEADLINE=${DEADLINE:-240}

# Fast enough to observe inside a test run, still many multiples of the renewal
# interval - so a live worker cannot be mistaken for a dead one.
export ORCH_REAPER_MIN_IDLE_S=${ORCH_REAPER_MIN_IDLE_S:-10}
export ORCH_REAPER_INTERVAL_S=${ORCH_REAPER_INTERVAL_S:-5}
export ORCH_LEASE_RENEW_INTERVAL_S=${ORCH_LEASE_RENEW_INTERVAL_S:-2}

banner() {
  echo
  echo "=============================================================="
  echo "  $1"
  echo "=============================================================="
}

reset_stack() {
  docker compose up -d --force-recreate --scale worker=3 worker api >/dev/null 2>&1
  sleep 6
  docker compose exec -T redis redis-cli FLUSHDB >/dev/null
  docker compose cp scripts/prove_recovery.py api:/tmp/prove.py >/dev/null 2>&1
  sleep 1
}

victim() {
  docker compose ps -q worker | head -1
}

run_scenario() {
  local mode=$1
  reset_stack

  docker compose exec -T api python /tmp/prove.py arm "$JOBS" "$PAGES" || return 1

  local target
  target=$(victim)
  local hostname_before
  hostname_before=$(docker exec "$target" hostname)
  echo "victim container ${target:0:12} (consumer name: $hostname_before)"

  # SIGKILL, not SIGTERM. SIGTERM runs the graceful drain, which finishes the
  # pages in flight and acks them - so it proves the shutdown path works and
  # says nothing about recovery. -9 is unblockable and leaves the PEL exactly as
  # the spec's "simulate worker node termination" intends.
  docker kill -s KILL "$target" >/dev/null
  echo "SIGKILLed. pending entries orphaned:"
  docker compose exec -T api python -c "
import json, urllib.request
d = json.load(urllib.request.urlopen('http://localhost:8000/queue/consumers'))
for lane, info in d['lanes'].items():
    for c in info['consumers']:
        print(f\"  {lane:<20} {c['name'][:12]}  pending={c['pending']}\")
"

  if [ "$mode" = restart ]; then
    # Same container, so the same hostname: read_own_pending territory.
    docker start "$target" >/dev/null
    echo "restarted SAME container -> consumer name unchanged -> own-pending path"
  else
    # New container, so a new hostname: the dead consumer never returns and
    # only the reaper can recover its entries.
    docker rm -f "$target" >/dev/null 2>&1
    docker compose up -d --scale worker=3 worker >/dev/null 2>&1
    echo "REPLACED the container -> new consumer name -> reaper path"
  fi

  docker compose exec -T api python /tmp/prove.py watch "$DEADLINE"
  local rc=$?

  # WHICH mechanism recovered the work, counted from the logs rather than
  # assumed. Without this the two scenarios are indistinguishable in their
  # output: both print "60/60 pages", and a reaper that silently did nothing
  # while own-pending quietly handled everything would look identical to a
  # working reaper. These counters are the difference between the demo proving
  # Step 14 and the demo proving Step 13 twice.
  echo "recovery attributed to:"
  echo "  resuming_own_pending  $(docker compose logs worker 2>/dev/null | grep -c resuming_own_pending)"
  echo "  orphans_claimed       $(docker compose logs worker 2>/dev/null | grep -c orphans_claimed)"
  echo "  orphan_requeued       $(docker compose logs worker 2>/dev/null | grep -c orphan_requeued)"
  echo "  orphan_already_term   $(docker compose logs worker 2>/dev/null | grep -c orphan_state_expired)"
  return $rc
}

banner "Scenario 1: worker RESTART (read_own_pending)"
run_scenario restart
restart_rc=$?

banner "Scenario 2: worker REPLACEMENT (the reaper)"
run_scenario replace
replace_rc=$?

banner "Result"
echo "worker restart     : $([ $restart_rc -eq 0 ] && echo PASS || echo FAIL)"
echo "worker replacement : $([ $replace_rc -eq 0 ] && echo PASS || echo FAIL)"

# Leave the stack in its shipped configuration, or every later measurement
# silently inherits this script's shortened reaper clocks.
unset ORCH_REAPER_MIN_IDLE_S ORCH_REAPER_INTERVAL_S ORCH_LEASE_RENEW_INTERVAL_S
docker compose up -d --force-recreate --scale worker=3 worker >/dev/null 2>&1

[ $restart_rc -eq 0 ] && [ $replace_rc -eq 0 ]
