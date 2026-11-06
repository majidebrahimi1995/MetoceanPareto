#!/usr/bin/env bash
# Invoked by submit_sge.sh; qsub exports PLATFORM_ROOT, PLATFORM_PYTHON, etc.
set -euo pipefail
cd "${PLATFORM_ROOT:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
case "${PLATFORM_STAGE:?}" in
  preflight|finalize|dense-report)
    exec "${PLATFORM_PYTHON:?}" platform.py pareto --stage "$PLATFORM_STAGE" \
      --centroids "$PLATFORM_CENTROIDS" --output "$PLATFORM_OUTPUT"
    ;;
  search)
    exec "${PLATFORM_PYTHON:?}" platform.py pareto --stage search \
      --centroids "$PLATFORM_CENTROIDS" --output "$PLATFORM_OUTPUT" \
      --task-id "${SGE_TASK_ID:?}" --workers "${NSLOTS:?}"
    ;;
  merge|dense)
    exec "${PLATFORM_PYTHON:?}" platform.py pareto --stage "$PLATFORM_STAGE" \
      --centroids "$PLATFORM_CENTROIDS" --output "$PLATFORM_OUTPUT" \
      --case "${SGE_TASK_ID:?}"
    ;;
  *) echo "Unknown stage: $PLATFORM_STAGE" >&2; exit 2 ;;
esac
