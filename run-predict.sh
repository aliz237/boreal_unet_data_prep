#!/usr/bin/env -S bash --login
set -euo pipefail
basedir=$(dirname "$(readlink -f "$0")")
mkdir -p output
conda run --live-stream --name data_prep python ${basedir}/predict.py \
      --hls_path ${1} \
      --topo_path ${2} \
      --out_raster_path ${3} \
      --patch_size ${4} \
      --step_size ${5} \
      --ndval ${6} \
      --batch_size ${7}

