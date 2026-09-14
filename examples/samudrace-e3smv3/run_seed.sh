#!/usr/bin/env bash
# Extract latents from a second rollout of SamudrACE-E3SMv3 with a different
# noise seed, for the stochastic-spread analysis (analysis.py spread).
#
#   examples/samudrace-e3smv3/run_seed.sh SEED [DATA_DIR]
#
# Requires run.sh to have been run first (it provides the environment, the
# download and extract-config.yaml). Writes latents_seed<SEED>/ and
# run_seed<SEED>_{ocean,atmosphere}/ next to the seed-0 outputs.
set -euo pipefail

SEED="${1:?usage: run_seed.sh SEED [DATA_DIR]}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DATA="$(cd "${2:-$REPO/demo_data/samudrace_e3smv3}" && pwd)"
PY="$DATA/.venv/bin/python"
cd "$DATA"

[ -f extract-config.yaml ] || { echo "run run.sh first: $DATA/extract-config.yaml is missing"; exit 1; }
sed -E "s/^seed: .*/seed: $SEED/" extract-config.yaml > "extract-config-seed$SEED.yaml"

export FME_USE_MPS=1 PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONWARNINGS=ignore

for component in ocean atmosphere; do
  short=$([ "$component" = atmosphere ] && echo atmos || echo ocean)
  out="latents_seed$SEED/$short"
  if [ -d "$out" ]; then echo "   $out exists, skipping"; continue; fi
  echo "== $component, seed $SEED"
  rm -rf run
  extra=()
  [ "$component" = atmosphere ] && extra=(--dtype float16)
  "$PY" "$REPO/scripts/extract_latents_fme.py" "extract-config-seed$SEED.yaml" \
    --component "$component" --out "$out" --max-times 4 "${extra[@]}"
  mv run "run_seed${SEED}_$component"
done
echo "== done: latents_seed$SEED/"
