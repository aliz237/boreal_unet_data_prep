#!/usr/bin/env -S bash --login
set -euo pipefail
basedir=$(dirname "$(readlink -f "$0")")
mkdir -p output
CMD=(conda run --live-stream --name predict_env python "${basedir}/predict_all_years.py"
      --tile_num "${1}"
      --stac_catalog "${2}"
      --model_path "${3}"
)

[[ -n "${4:-}" ]] && CMD+=(--output_dir "$4")
[[ -n "${5:-}" ]] && CMD+=(--input_dir "$5")
[[ -n "${6:-}" ]] && CMD+=(--patch_size "$6")
[[ -n "${7:-}" ]] && CMD+=(--step_size "$7")
[[ -n "${8:-}" ]] && CMD+=(--ndval "${8}")
[[ -n "${9:-}" ]] && CMD+=(--batch_size "${9}")
[[ -n "${10:-}" ]] && CMD+=(--max_na_block "${10}")
[[ -n "${11:-}" ]] && CMD+=(--nodata_thresh "${11}")
[[ -n "${12:-}" ]] && CMD+=(--agb)

echo "${CMD[@]}"
"${CMD[@]}"
