#!/usr/bin/env bash
# Evaluate one or more candidates of a finished run on the held-out test set.
#
#   bash scripts/eval_extra_candidate.sh <run_name> <candidate> [<candidate> ...]
#
# scripts/search.sh calls this for the selected winner. It is also useful on its
# own to test a candidate that selection did not pick.
#
# At the end of a run the candidate files are archived into logs/<run>/agents/
# and removed from the shared agents/ directory, so that runs cannot contaminate
# each other. A run started with --snapshot-code also has a self-contained copy
# of the code under logs/<run>/code/text_classification/. This script copies the
# candidate back into that snapshot and runs the test evaluation entirely inside
# it, touching neither the shared agents/ directory nor any other running job.
#
# Requires the run to have been started with --snapshot-code, and the memory
# states written during validation (memory.json) to still be present for every
# dataset; without them the test evaluation fails.
#
# FORCE=1 deletes any existing test.json first. benchmark.py --test skips a
# candidate whose test.json already exists and just reprints it, so re-running a
# contaminated evaluation needs this.

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: bash scripts/eval_extra_candidate.sh <run_name> <candidate> [<candidate> ...]"
  exit 1
fi

RUN_NAME="$1"; shift
CANDIDATES=("$@")

PROJECT_ROOT="$(pwd)"
LOGS_DIR="$PROJECT_ROOT/logs/$RUN_NAME"
SNAP="$LOGS_DIR/code/text_classification"
ARCHIVE="$LOGS_DIR/agents"
RESULTS_DIR="$LOGS_DIR/results"

if [ ! -d "$SNAP" ]; then
  echo "ERROR: no code snapshot at $SNAP; this run was not started with --snapshot-code"
  exit 1
fi

# Copy the candidate from the archive into the snapshot, if it is not there yet
for name in "${CANDIDATES[@]}"; do
  dst="$SNAP/agents/$name.py"
  if [ ! -f "$dst" ]; then
    src="$ARCHIVE/$name.py"
    if [ -f "$src" ]; then
      cp "$src" "$dst"
      echo "copied $name.py: archive -> snapshot"
    else
      echo "ERROR: $name.py is in neither $ARCHIVE nor $SNAP/agents"
      exit 1
    fi
  fi
done

# FORCE=1: delete any existing test.json to force a re-evaluation
if [[ "${FORCE:-0}" == "1" ]]; then
    for name in "${CANDIDATES[@]}"; do
        for f in "$RESULTS_DIR"/*/"$name"/*/test.json; do
            [[ -f "$f" ]] && rm "$f" && echo "removed (force): $f"
        done
    done
fi

# Run the test evaluation inside the snapshot only, the same way the search does
for name in "${CANDIDATES[@]}"; do
  echo
  echo "=== test evaluation: $name ==="
  (cd "$SNAP" && uv run --project "$PROJECT_ROOT" python benchmark.py \
    --memory "$name" --test \
    --logs-dir "$LOGS_DIR" --results-dir "$RESULTS_DIR")
done

# Summary
echo
echo "=== summary ==="
for name in "${CANDIDATES[@]}"; do
  python3 -c "
import json, glob
accs = []
for f in glob.glob('$RESULTS_DIR/*/$name/*/test.json'):
    d = json.load(open(f))
    if d.get('accuracy') is not None:
        accs.append((f.split('/')[-4], round(d['accuracy']*100, 1)))
avg = sum(a for _, a in accs) / len(accs) if accs else None
print('$name:', accs, 'avg=', round(avg, 1) if avg else None)
"
done
