#!/usr/bin/env bash
# Poll the DETACHED Lambda job from any session, anytime.
#   ./lambda_poll.sh              # show remote status + pull outputs -> lambda_pull/
#   ./lambda_poll.sh --terminate  # ...and terminate the instance if you're done
# Reads .lambda/current.env (written at launch). Safe to re-run.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
HERE="$(pwd)"
SCRIPTS="$HERE/../lambda_local/scripts"

source ./lambda.env
[ -f .lambda/current.env ] && source .lambda/current.env
: "${LAMBDA_HOST:?no LAMBDA_HOST - launch first}"
: "${LAMBDA_INSTANCE_ID:?no LAMBDA_INSTANCE_ID}"
REMOTE_PROJECT="/lambda/nfs/${LAMBDA_FS}/projects/$(basename "$HERE")"
SSH_OPTS="-o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -o BatchMode=yes"

echo "=== instance ${LAMBDA_INSTANCE_ID} @ ${LAMBDA_HOST} ($(date -u +%FT%TZ)) ==="
ssh $SSH_OPTS ubuntu@"$LAMBDA_HOST" "
  cd '$REMOTE_PROJECT' 2>/dev/null || { echo 'project dir not found on NFS'; exit 0; }
  echo '--- status.txt ---';        tail -10 outputs/status.txt 2>/dev/null
  echo '--- heartbeat (last 2) ---';tail -2  outputs/heartbeat.log 2>/dev/null
  echo '--- train tail ---';        tail -4  outputs/train_ml100k.log 2>/dev/null | tr -d '\r'
  echo '--- metrics ---';           cat      outputs/metrics.txt 2>/dev/null
  echo '--- DONE? ---'; [ -f outputs/DONE ] && echo \"DONE rc=\$(cat outputs/DONE)\" || echo 'still running (no DONE marker)'
" 2>&1 || echo "(ssh failed; the instance may already be terminated)"

echo "=== pull outputs -> lambda_pull/ ==="
"$SCRIPTS/lambda_sync.sh" pull || echo "(pull failed / nothing yet)"

if [ "${1:-}" = "--terminate" ]; then
  echo "=== terminate ${LAMBDA_INSTANCE_ID} ==="
  curl -sS -X POST "https://cloud.lambda.ai/api/v1/instance-operations/terminate" \
    -H "Authorization: Bearer ${LAMBDA_API_KEY}" -H "Content-Type: application/json" \
    -d "{\"instance_ids\":[\"${LAMBDA_INSTANCE_ID}\"]}" \
    | jq -r '.data.terminated_instances[]? | "terminated: \(.id) -> \(.status)"' 2>/dev/null \
    || echo "(terminate sent)"
fi
