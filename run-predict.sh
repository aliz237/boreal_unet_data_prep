#!/usr/bin/env -S bash --login
set -euo pipefail
basedir=$(dirname "$(readlink -f "$0")")
mkdir -p output
conda run --live-stream --name predict_env python ${basedir}/predict.py \
      --tile_num ${1} \
      --year ${2} \
      --stac_catalog ${3} \
      --model_path ${4} \
      --out_raster_path ${5} \
      --patch_size ${6} \
      --step_size ${7} \
      --ndval ${8} \
      --batch_size ${9}

