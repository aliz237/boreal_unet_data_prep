#!/usr/bin/env -S bash --login
set -euo pipefail
basedir=$(dirname "$(readlink -f "$0")")
mkdir -p output
conda run --live-stream --name data_prep python ${basedir}/data_prep.py \
      --tile_num ${1} \
      --year ${2} \
      --hls_path ${3} \
      --slope_path ${4} \
      --atl08_path ${5} \
      --patch_size ${6} \
      --overlap ${7}
