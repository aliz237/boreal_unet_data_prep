#!/usr/bin/env -S bash --login
set -euo pipefail
basedir=$(dirname "$(readlink -f "$0")")
mkdir -p output
conda run --live-stream --name data_prep python ${basedir}/predict.py \
      --hls_path ${1} \
      --topo_path ${2} \
      --lc_path ${3} \
      --model_path ${4} \
      --out_raster_path ${5} \
      --patch_size ${6} \
      --step_size ${7} \
      --ndval ${8} \
      --batch_size ${9}

