#!/usr/bin/env bash
# Phase 0 ACPC diagnostics — batch runner for Paper 1
# Usage:
#   bash run_phase0_acpc.sh              # full 72 ckpts (LeWM + PLDM)
#   bash run_phase0_acpc.sh --dry-run    # resolve manifests only
#   bash run_phase0_acpc.sh --single     # single checkpoint dry-run (PushT LeWM 0.0)
#   bash run_phase0_acpc.sh --lewm-only  # LeWM 36 ckpts only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EVAL_LEWM="assets/paper1_data/canonical_evals_20260517.json"
EVAL_PLDM="assets/paper1_data/canonical_evals_pldm_20260522.json"
LOCAL_EVAL_LEWM="/tmp/canonical_evals_local.json"
LOCAL_EVAL_PLDM="/tmp/canonical_evals_pldm_local.json"
OUT="assets/paper1_data/acpc_phase0_diagnostics.json"

CANONICAL_DATA_ROOT="${CANONICAL_DATA_ROOT:-}"
if [[ -z "$CANONICAL_DATA_ROOT" ]]; then
    CANONICAL_DATA_ROOT="$(python3 - "$EVAL_LEWM" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for task_block in data.values():
    if isinstance(task_block, dict):
        for entry in task_block.values():
            path = entry.get("path") if isinstance(entry, dict) else None
            if path and "/ckpt/" in path:
                print(path.split("/ckpt/", 1)[0].rsplit("/", 1)[0])
                raise SystemExit
raise SystemExit("could not infer canonical data root from eval manifest")
PY
)"
fi
DATA_ROOT="${DATA_ROOT:-${STABLEWM_HOME:-$CANONICAL_DATA_ROOT}}"

# ---------------------------------------------------------------------------
# Prepare localized eval JSONs when the desired data prefix differs from the
# prefix embedded in the canonical eval manifests.
# ---------------------------------------------------------------------------
echo "[phase0] Preparing localized eval manifests..."
if [[ "$DATA_ROOT" == "$CANONICAL_DATA_ROOT" ]]; then
    LOCAL_EVAL_LEWM="$EVAL_LEWM"
    LOCAL_EVAL_PLDM="$EVAL_PLDM"
    echo "[phase0]   -> using canonical eval manifests"
else
    if [[ ! -f "$LOCAL_EVAL_LEWM" ]] || [[ "$(stat -c %Y "$EVAL_LEWM")" -gt "$(stat -c %Y "$LOCAL_EVAL_LEWM" 2>/dev/null || echo 0)" ]]; then
        sed "s|$CANONICAL_DATA_ROOT|$DATA_ROOT|g" "$EVAL_LEWM" > "$LOCAL_EVAL_LEWM"
        echo "[phase0]   -> $LOCAL_EVAL_LEWM"
    fi
    if [[ ! -f "$LOCAL_EVAL_PLDM" ]] || [[ "$(stat -c %Y "$EVAL_PLDM")" -gt "$(stat -c %Y "$LOCAL_EVAL_PLDM" 2>/dev/null || echo 0)" ]]; then
        sed "s|$CANONICAL_DATA_ROOT|$DATA_ROOT|g" "$EVAL_PLDM" > "$LOCAL_EVAL_PLDM"
        echo "[phase0]   -> $LOCAL_EVAL_PLDM"
    fi
fi

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export STABLEWM_HOME="$DATA_ROOT"

# Verify deps
echo "[phase0] Verifying environment..."
if [[ "${1:-}" == "--dry-run" || "${1:-}" == "--single" ]]; then
    echo "  SKIP (dry-run does not load models)"
elif ! python3 -c "import torch; import stable_worldmodel; import stable_pretraining; print('  OK')" 2>/dev/null; then
    echo "[phase0] ERROR: torch / stable_worldmodel / stable_pretraining not found."
    echo "[phase0] Please activate the correct conda/venv."
    exit 1
fi

# ---------------------------------------------------------------------------
# Run mode selection
# ---------------------------------------------------------------------------
METHODS="LeWM PLDM"
TASKS="TwoRoom PushT Reacher Cube"
STD_KEYS="0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08"
LIMIT=""

case "${1:-}" in
    --dry-run)
        echo "[phase0] DRY-RUN mode: resolving manifests and model files only."
        OUT="/tmp/acpc_phase0_dry_run.json"
        EXTRA_FLAGS="--dry-run"
        ;;
    --single)
        echo "[phase0] SINGLE mode: PushT LeWM 0.0 dry-run only."
        METHODS="LeWM"
        TASKS="PushT"
        STD_KEYS="0.0"
        OUT="/tmp/acpc_phase0_single.json"
        EXTRA_FLAGS="--dry-run"
        ;;
    --lewm-only)
        echo "[phase0] LEWM-ONLY mode: 36 ckpts."
        METHODS="LeWM"
        ;;
    *)
        echo "[phase0] FULL mode: LeWM + PLDM (72 ckpts). This will take 1–3 hours on GPU."
        EXTRA_FLAGS=""
        ;;
esac

# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
echo "[phase0] Starting at $(date -Iseconds)"
echo "[phase0] Output: $OUT"

python3 -m tools.paper1_phase0_acpc \
    --methods $METHODS \
    --tasks $TASKS \
    --std-keys $STD_KEYS \
    --evals-lewm "$LOCAL_EVAL_LEWM" \
    --evals-pldm "$LOCAL_EVAL_PLDM" \
    --out "$OUT" \
    ${LIMIT:-} \
    ${EXTRA_FLAGS:-}

echo "[phase0] Finished at $(date -Iseconds)"
echo "[phase0] Output: $OUT"

# Quick status report
python3 -c "
import json, sys
with open('$OUT') as f:
    d = json.load(f)
rows = d['rows']
ok = sum(1 for r in rows if r['status'] == 'ok')
err = sum(1 for r in rows if r['status'] == 'error')
dry = sum(1 for r in rows if r['status'] == 'dry_run')
print(f'[phase0] Results: {ok} OK, {err} error, {dry} dry-run, {len(rows)} total')
"
