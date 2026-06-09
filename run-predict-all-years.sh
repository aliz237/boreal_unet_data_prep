#!/usr/bin/env -S bash --login
set -euo pipefail
basedir=$(dirname "$(readlink -f "$0")")
export MPLBACKEND="Agg"
AGB=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tile_num)   TILE_NUM="$2"; shift 2 ;;
    --hls_tindex) HLS_TINDEX="$2"; shift 2 ;;
    --topo_path)  TOPO_PATH="$2"; shift 2 ;;
    --lc_path)    LC_PATH="$2"; shift 2 ;;
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --patch_size) PATCH_SIZE="$2"; shift 2 ;; # Defaults at 128
    --step_size)    STEP_SIZE="$2"; shift 2 ;; # Default of 32
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --input_dir)  INPUT_DIR="$2"; shift 2 ;;
    --nodata_thresh) NODATA_THRESH="$2"; shift 2 ;;
    --max_na_block) MAX_NA_BLOCK="$2"; shift 2 ;;
    --agb)        AGB=true; shift 1 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "${OUTPUT_DIR:-}" ]]; then
    OUTPUT_DIR="output"
    mkdir -p output
fi

if [[ -z "${INPUT_DIR:-}" ]]; then
    INPUT_DIR="input"
    mkdir -p input
fi


CMD=(
  conda run --live-stream --name predict_env python "${basedir}/predict_all_years.py"
  --hls_tindex "$HLS_TINDEX"
  --tile_num "$TILE_NUM"
  --topo_path "$TOPO_PATH"
  --lc_path "$LC_PATH"
  --model_path "$MODEL_PATH"
  --output_dir "$OUTPUT_DIR"
  --input_dir "$INPUT_DIR"
)

if [[ -n "${PATCH_SIZE:-}" ]]; then
    CMD+=(--patch_size "$PATCH_SIZE")
fi

if [[ -n "${OVERLAP:-}" ]]; then
    CMD+=(--overlap "$OVERLAP")
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

echo "${CMD[@]}"
"${CMD[@]}"
