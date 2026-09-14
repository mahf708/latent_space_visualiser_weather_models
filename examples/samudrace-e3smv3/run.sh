#!/usr/bin/env bash
# Run the latent space visualiser on real SamudrACE-E3SMv3 latents.
#
#   examples/samudrace-e3smv3/run.sh [DATA_DIR]
#
# Downloads the checkpoint, initial conditions and forcing from HuggingFace
# (about 2.7 GB), extracts ocean and atmosphere latents from a short rollout,
# builds reference fields from the same rollout's predictions, and starts the
# app. Each stage is skipped when its output already exists, so it is safe to
# re-run. Needs uv (https://docs.astral.sh/uv/). On Apple silicon the rollout
# runs on the GPU; elsewhere it falls back to CPU, which is slower.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DATA="$(mkdir -p "${1:-$REPO/demo_data/samudrace_e3smv3}" && cd "${1:-$REPO/demo_data/samudrace_e3smv3}" && pwd)"
PY="$DATA/.venv/bin/python"
HF=https://huggingface.co/allenai/SamudrACE-E3SMv3/resolve/main

echo "== environment"
if [ ! -x "$PY" ]; then
  uv venv --python 3.12 "$DATA/.venv"
  VIRTUAL_ENV="$DATA/.venv" uv pip install \
    "fme @ git+https://github.com/mahf708/ace@main" -r "$REPO/requirements.txt"
fi

echo "== download"
mkdir -p "$DATA/hf/forcing_data" "$DATA/hf/initial_conditions"
for f in \
  initial_conditions/SamudrACE-E3SMv3-ICx3-train_ocean_ic.nc \
  initial_conditions/SamudrACE-E3SMv3-ICx3-train_atmosphere_ic.nc \
  forcing_data/ocean-forcing-1yr.nc \
  forcing_data/atmosphere-forcing-1yr.nc \
  SamudrACE-E3SMv3.tar; do
  if [ -f "$DATA/hf/$f" ]; then echo "   $f present"; continue; fi
  # resume into a .part file; a completed file is only renamed into place once
  curl -fSL -C - --retry 5 -o "$DATA/hf/$f.part" "$HF/$f"
  mv "$DATA/hf/$f.part" "$DATA/hf/$f"
done

sed "s|DATA|$DATA|g" "$HERE/extract-config.yaml" > "$DATA/extract-config.yaml"
cp "$HERE/paths.json" "$DATA/paths.json"

export FME_USE_MPS=1 PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONWARNINGS=ignore
cd "$DATA"

extract() {  # component, output dir, extra args...
  local component=$1 out=$2; shift 2
  if [ -d "$out" ]; then echo "   $out exists, skipping"; return; fi
  rm -rf run
  "$PY" "$REPO/scripts/extract_latents_fme.py" extract-config.yaml \
    --component "$component" --out "$out" --max-times 4 "$@"
  mv run "run_$component"
}

echo "== ocean latents (9 U-Net levels, 5-day steps)"
extract ocean latents/ocean --write-grid ocean_grid.npz

echo "== atmosphere latents (8 SFNO blocks, 6-hour steps)"
extract atmosphere latents/atmos --write-grid atmos_grid.npz --dtype float16

echo "== reference fields from the same rollout"
mkdir -p reference
[ -f reference/ocean_reference.nc ] || "$PY" "$REPO/scripts/fme_predictions_to_reference.py" \
  run_ocean/ocean reference/ocean_reference.nc --year-offset 1600
[ -f reference/atmos_reference.nc ] || "$PY" "$REPO/scripts/fme_predictions_to_reference.py" \
  run_atmosphere/atmosphere reference/atmos_reference.nc --year-offset 1600 --max-times 8

if [ -n "${NO_APP:-}" ]; then echo "== ready (NO_APP set, not starting the app)"; exit 0; fi
echo "== app"
cp -R "$REPO/.streamlit" "$DATA/" 2>/dev/null || true   # the app's theme
exec "$DATA/.venv/bin/streamlit" run "$REPO/app.py"
