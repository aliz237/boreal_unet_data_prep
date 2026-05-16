#!/usr/bin/env -S bash --login
set -euo pipefail
basedir=$(dirname "$(readlink -f "$0")")
mkdir -p output

ATL08AGB="output/ATL08_AGB_${1}_${2}.parquet"

# transform ICESat2 atl08 RH metrics to AGB density
# using packaged OLS models (-b) fitted on plot level data
conda run --live-stream --name data_prep Rscript ${basedir}/atl08_to_agb.R \
    -a ${5} \
    -b ${9} \
    -o ${ATL08AGB}

# Then use ICESat2 AGB (and RH metrics) as target variable in tfrecords
conda run --live-stream --name data_prep python ${basedir}/data_prep.py \
      --tile_num ${1} \
      --year ${2} \
      --hls_path ${3} \
      --topo_path ${4} \
      --atl08_path ${ATL08AGB} \
      --patch_size ${6} \
      --overlap ${7} \
      --rh ${8} \
      --fire_path "${10:-}"
