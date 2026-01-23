#!/usr/bin/env -S bash --login
set -euo pipefail
basedir=$(dirname "$(readlink -f "$0")")
mkdir -p output
conda run --live-stream --name data_prep python ${basedir}/predict.py \
      --hls_path ${1} \
      --topo_path ${2} \
      --model_path ${3} \
      --out_raster_path ${4} \
      --patch_size ${5} \
      --step_size ${6} \
      --ndval ${7} \
      --batch_size ${8}

