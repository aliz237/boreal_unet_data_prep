#!/usr/bin/env -S bash --login
set -euo pipefail
basedir=$(dirname "$(readlink -f "$0")")
export MPLBACKEND="Agg"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tile_num)   TILE_NUM="$2"; shift 2 ;;
    --year)       YEAR="$2"; shift 2 ;;
    --hls_path)   HLS_PATH="$2"; shift 2 ;;
    --atl08_path) ATL08_PATH="$2"; shift 2 ;;
    --atl08_agb_path) ATL08_AGB_PATH="$2"; shift 2 ;;
    --topo_path)  TOPO_PATH="$2"; shift 2 ;;
    --bio_models) BIO_MODELS_PATH="$2"; shift 2 ;; # Everything else below is optional
    --patch_size) PATCH_SIZE="$2"; shift 2 ;; # Defaults at 128
    --overlap)    OVERLAP="$2"; shift 2 ;; # Default of 32
    --rh)         RH="$2"; shift 2 ;; # Default of RH98
    --fire_path)  FIRE_PATH="$2"; shift 2 ;;
    --out_dir)    OUT_DIR="$2"; shift 2 ;;
    --agb)        AGB=true; shift 1 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "${OUT_DIR:-}" ]]; then
    OUT_DIR="output"
    mkdir -p output
fi

# first transform ICESat2 atl08 RH metrics to AGB density
# using packaged OLS models (-b) fitted on plot level data
RCMD=(
    conda run --live-stream --name data_prep2 Rscript "${basedir}/atl08_to_agb.R"
    -a "$ATL08_PATH"
    -b "$BIO_MODELS_PATH"
    -o "${ATL08_AGB_PATH}"
)
echo "${RCMD[@]}"
"${RCMD[@]}"

# Then use ICESat2 AGB (and RH metrics) as target variable in tfrecords
CMD=(
  conda run --live-stream --name data_prep2 python "${basedir}/data_prep.py"
  --tile_num "$TILE_NUM"
  --year "$YEAR"
  --hls_path "$HLS_PATH"
  --topo_path "$TOPO_PATH"
  --atl08_path "$ATL08_AGB_PATH"
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

if [[ -n "${AGB:-}" ]]; then
    CMD+=(--agb)
fi

echo "${CMD[@]}"
"${CMD[@]}"
