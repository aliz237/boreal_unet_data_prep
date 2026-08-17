#!/usr/bin/env -S bash --login
set -euo pipefail
basedir=$(dirname "$(readlink -f "$0")")
export MPLBACKEND="Agg"
AGB=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tile_num)     TILE_NUM="$2"; shift 2 ;;
    --stac_catalog) STAC_CATALOG="$2"; shift 2 ;; # Everything else below is optional
    --patch_size)   PATCH_SIZE="$2"; shift 2 ;; # Defaults at 128
    --overlap)      OVERLAP="$2"; shift 2 ;; # Default of 32
    --rh)           RH="$2"; shift 2 ;; # Default of h_canopy
    --fire_path)    FIRE_PATH="$2"; shift 2 ;;
    --out_dir)      OUT_DIR="$2"; shift 2 ;;
    --ndval_thresh) NDVAL_THRESH="$2"; shift 2 ;;
    --agb)          AGB=true; shift 1 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "${OUT_DIR:-}" ]]; then
    OUT_DIR="output"
    mkdir -p output
fi

# atl08_to_agb.R is invoked internally by data_prep.py (via atl08_to_raster) when --agb is passed
CMD=(
  conda run --live-stream --name data_prep2 python "${basedir}/data_prep.py"
  --tile_num "$TILE_NUM"
  --stac_catalog "$STAC_CATALOG"
  --out_dir "$OUT_DIR"
)

if [[ -n "${PATCH_SIZE:-}" ]]; then
    CMD+=(--patch_size "$PATCH_SIZE")
fi

if [[ -n "${OVERLAP:-}" ]]; then
    CMD+=(--overlap "$OVERLAP")
fi

if [[ -n "${RH:-}" ]]; then
    CMD+=(--rh "$RH")
fi

if [[ -n "${FIRE_PATH:-}" ]]; then
  CMD+=(--fire_path "$FIRE_PATH")
fi

if [[ -n "${NDVAL_THRESH:-}" ]]; then
    CMD+=(--ndval_thresh "$NDVAL_THRESH")
fi

if [[ "${AGB}" == true ]]; then
    CMD+=(--agb)
fi

echo "${CMD[@]}"
"${CMD[@]}"
