#!/usr/bin/env -S bash --login
set -euo pipefail
basedir=$(dirname "$(readlink -f "$0")")
export MPLBACKEND="Agg"
AGB=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tile_num)         TILE_NUM="$2"; shift 2 ;; # comma-separated, e.g. "3364,3365"
    --year)              YEAR="$2"; shift 2 ;;
    --stac_catalog)      STAC_CATALOG="$2"; shift 2 ;;
    --model_path)        MODEL_PATH="$2"; shift 2 ;;
    --out_raster_path)   OUT_RASTER_PATH="$2"; shift 2 ;;
    --input_dir)         INPUT_DIR="$2"; shift 2 ;;
    --patch_size)        PATCH_SIZE="$2"; shift 2 ;; # Defaults at 128
    --step_size)         STEP_SIZE="$2"; shift 2 ;; # Default of 100
    --ndval)             NDVAL="$2"; shift 2 ;;
    --batch_size)        BATCH_SIZE="$2"; shift 2 ;;
    --nodata_thresh)     NODATA_THRESH="$2"; shift 2 ;;
    --max_na_block)      MAX_NA_BLOCK="$2"; shift 2 ;;
    --n_threads)         N_THREADS="$2"; shift 2 ;;
    --agb)                AGB=true; shift 1 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "${INPUT_DIR:-}" ]]; then
    INPUT_DIR="input"
    mkdir -p input
fi

# predict.py's --tile_num takes one or more space-separated values; split the
# comma-separated string this wrapper accepts into that form.
IFS=',' read -ra TILE_NUMS <<< "$TILE_NUM"

CMD=(
  conda run --live-stream --name predict_env python "${basedir}/predict.py"
  --tile_num "${TILE_NUMS[@]}"
  --year "$YEAR"
  --stac_catalog "$STAC_CATALOG"
  --model_path "$MODEL_PATH"
  --out_raster_path "$OUT_RASTER_PATH"
  --input_dir "$INPUT_DIR"
)

if [[ -n "${PATCH_SIZE:-}" ]]; then
    CMD+=(--patch_size "$PATCH_SIZE")
fi

if [[ -n "${STEP_SIZE:-}" ]]; then
    CMD+=(--step_size "$STEP_SIZE")
fi

if [[ -n "${NDVAL:-}" ]]; then
    CMD+=(--ndval "$NDVAL")
fi

if [[ -n "${BATCH_SIZE:-}" ]]; then
    CMD+=(--batch_size "$BATCH_SIZE")
fi

if [[ "${AGB}" == true ]]; then
    CMD+=(--agb)
fi

if [[ -n "${MAX_NA_BLOCK:-}" ]]; then
    CMD+=(--max_na_block "$MAX_NA_BLOCK")
fi

if [[ -n "${NODATA_THRESH:-}" ]]; then
    CMD+=(--nodata_thresh "$NODATA_THRESH")
fi

if [[ -n "${N_THREADS:-}" ]]; then
    CMD+=(--n_threads "$N_THREADS")
fi

echo "${CMD[@]}"
"${CMD[@]}"
