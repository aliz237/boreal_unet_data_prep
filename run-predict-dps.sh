#!/usr/bin/env -S bash --login
set -euo pipefail
basedir=$(dirname "$(readlink -f "$0")")
mkdir -p output
CMD=(conda run --live-stream --name predict_env python "${basedir}/predict.py"
      --tile_num "${1}"
      --year "${2}"
      --stac_catalog "${3}"
      --model_path "${4}"
      --out_raster_path "${5}"
)

[[ -n "${6:-}" ]] && CMD+=(--input_dir "$6")
[[ -n "${7:-}" ]] && CMD+=(--patch_size "$7")
[[ -n "${8:-}" ]] && CMD+=(--step_size "$8")
[[ -n "${9:-}" ]] && CMD+=(--ndval "${9}")
[[ -n "${10:-}" ]] && CMD+=(--batch_size "${10}")
[[ -n "${11:-}" ]] && CMD+=(--max_na_block "${11}")
[[ -n "${12:-}" ]] && CMD+=(--nodata_thresh "${12}")
[[ -n "${13:-}" ]] && CMD+=(--agb)

echo "${CMD[@]}"
"${CMD[@]}"
