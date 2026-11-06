#!/usr/bin/env bash
# Usage: bash scripts/submit_sge.sh QUEUE OUTPUT [CENTROIDS] [SLOTS=16] [CONCURRENT=4]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUEUE="${1:?Specify SGE queue, e.g. meccsys1.q@node-m-9.local}"
OUTPUT_ARG="${2:?Specify a new output directory}"
CENTROIDS_ARG="${3:-$ROOT/optimization_pareto/centroid_pilot_v2/config/centroids_validation_exact_wd167p830.csv}"
SLOTS="${4:-16}"
CONCURRENT="${5:-4}"
PLATFORM_PYTHON="${PLATFORM_PYTHON:-$(command -v python)}"
PLATFORM_PYTHON="$(realpath "$PLATFORM_PYTHON")"
PLATFORM_ROOT="$ROOT"
PLATFORM_OUTPUT="$(realpath -m "$OUTPUT_ARG")"
PLATFORM_CENTROIDS="$(realpath "$CENTROIDS_ARG")"
export PLATFORM_PYTHON PLATFORM_ROOT PLATFORM_OUTPUT PLATFORM_CENTROIDS
test -f "$PLATFORM_CENTROIDS"
command -v qsub >/dev/null
mkdir -p "$ROOT/logs"
cd "$ROOT"
COMMON=(-terse -cwd -V -q "$QUEUE" -o "$ROOT/logs" -e "$ROOT/logs")
submit() {
  local stage="$1" hold="$2" slots="$3" range="$4" throttle="$5"
  local cmd=(qsub "${COMMON[@]}" -N "MP_${stage//-/_}" -pe smp "$slots"
             -v "PLATFORM_STAGE=$stage")
  if [[ -n "$hold" ]]; then cmd+=(-hold_jid "$hold"); fi
  if [[ -n "$range" ]]; then cmd+=(-t "$range" -tc "$throttle"); fi
  cmd+=("$ROOT/scripts/sge_stage.sh")
  "${cmd[@]}"
}
pre="$(submit preflight '' 1 '' '')"
search="$(submit search "$pre" "$SLOTS" 1-30 "$CONCURRENT")"
merge="$(submit merge "$search" 1 1-10 "$CONCURRENT")"
final="$(submit finalize "$merge" 1 '' '')"
dense="$(submit dense "$final" 1 1-10 "$CONCURRENT")"
report="$(submit dense-report "$dense" 1 '' '')"
printf 'Submitted: preflight=%s search=%s merge=%s finalize=%s dense=%s dense-report=%s\n' \
  "$pre" "$search" "$merge" "$final" "$dense" "$report"
printf 'Results: %s/dense_budget_0to10\n' "$PLATFORM_OUTPUT"
