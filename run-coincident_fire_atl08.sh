#!/usr/bin/env -S bash --login
set -euo pipefail
basedir=$(dirname "$(readlink -f "$0")")
mkdir -p output
conda run --live-stream --name data_prep python ${basedir}/coincident_fire_atl08.py \
      --tile_num ${1} \
      --fire_path ${2} \
      --atl08_years ${3} \
      --atl08_paths ${4} \
      --hls_path ${5}
