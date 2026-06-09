#!/usr/bin/env -S bash --login
set -euo pipefail
basedir=$(dirname "$(readlink -f "$0")")
mkdir -p output
CMD=(conda run --live-stream --name predict_env python "${basedir}/predict_all_years.py"
      --tile_num "${1}"
      --hls_tindex "${2}"
      --topo_path "${3}"
      --lc_path "${4}"
      --model_path "${5}"
)

[[ -n "${6:-}" ]] && CMD+=(--output_dir "$6")
[[ -n "${7:-}" ]] && CMD+=(--input_dir "$7")
[[ -n "${8:-}" ]] && CMD+=(--patch_size "$8")
[[ -n "${9:-}" ]] && CMD+=(--step_size "$9")
[[ -n "${10:-}" ]] && CMD+=(--ndval "${10}")
[[ -n "${11:-}" ]] && CMD+=(--batch_size "${11}")
[[ -n "${12:-}" ]] && CMD+=(--max_na_block "${12}")
[[ -n "${13:-}" ]] && CMD+=(--nodata_thresh "${13}")
[[ -n "${14:-}" ]] && CMD+=(--agb)

echo "${CMD[@]}"
"${CMD[@]}"
