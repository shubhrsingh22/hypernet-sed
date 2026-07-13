#!/usr/bin/env bash
# Wait for a running sweep (given its PID) to finish, then launch the
# bidirectional HCRNN-128 sweep. Chaining off the exact PID avoids the
# self-matching pgrep pitfall.
#
# Usage: scripts/chain_bi.sh <pid_to_wait_for> [seeds] [experiments] [gpus]
set -uo pipefail

WAIT_PID="${1:?need a PID to wait for}"
SEEDS_ARG="${2:-42 1337 2024}"
EXPS_ARG="${3:-hcrnn128_bi}"
GPUS_ARG="${4:-0 1 2 3}"

echo "[chain] waiting for runner PID ${WAIT_PID} to finish ..."
while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 60; done
echo "[chain] previous sweep finished; launching: ${EXPS_ARG} seeds=[${SEEDS_ARG}]"

exec bash scripts/run_seeds.sh "${SEEDS_ARG}" "${EXPS_ARG}" "${GPUS_ARG}"
