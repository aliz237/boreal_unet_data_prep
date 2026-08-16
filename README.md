# boreal-unet-data-prep

Training-data extraction and inference pipeline for predicting boreal forest
canopy height and aboveground biomass (AGB) from HLS imagery, ICESat-2 ATL08
LiDAR, and topography, using a U-Net. Built to run at tile-by-tile scale, both
locally and as registered algorithms on NASA MAAP's DPS job runner.

## Pipeline stages

1. **`coincident_fire_atl08.py`** *(optional)* -- finds fire polygons that
   have both pre- and post-fire ATL08 LiDAR coverage, for building
   fire-disturbance training/validation sets.
2. **`data_prep.py`** -- fuses HLS + ATL08 + topography into normalized,
   patch-extracted TFRecords for training. Internally shells out to
   **`atl08_to_agb.R`** to convert ATL08 height metrics to AGB when `--agb`
   is passed.
3. **Model training** -- currently done in a Jupyter notebook that isn't
   checked into this repo yet (an updated version is coming).
4. **`predict.py`** / **`predict_all_years.py`** -- run a trained U-Net over
   a single HLS scene, or over every available year for a tile.

Shared logic lives in `constants.py`, `raster_utils.py`, `atl08_utils.py`,
`patch_extraction.py`, and `tfrecord_utils.py` -- `data_prep.py` and
`predict.py` both import from these rather than from each other.

## Setup

This repo uses **two conda environments**, split mainly because
training-data extraction needs R (for `atl08_to_agb.R`) and inference
doesn't. Both pull CUDA-enabled TensorFlow (`tensorflow[and-cuda]`) for
consistency, even though DPS's current predict workers are CPU-only --
`tensorflow[and-cuda]` falls back to CPU cleanly when no GPU is present, so
this keeps the two envs interchangeable and ready for a GPU predict worker
without an environment change:

| Env | Created from | Used by |
|---|---|---|
| `data_prep2` | `environment.yml` | `data_prep.py` |
| `predict_env` | `environment-predict.yml` | `predict.py`, `predict_all_years.py`, `coincident_fire_atl08.py` |

```bash
conda env create -f environment.yml          # creates data_prep2
conda env create -f environment-predict.yml  # creates predict_env
# or, equivalently, for predict_env:
./build-env-predict.sh
```

GDAL/rasterio/geopandas are installed via conda-forge (not pip) so that
Python and R share one `libgdal` build -- see `pyproject.toml` for why its
`dependencies` list deliberately excludes GDAL and leaves versions unpinned.

### Local development (tests, linting)

Activate one of the conda envs above (either works; `pip` will pull in
whichever of `pyproject.toml`'s dependencies conda didn't already provide --
e.g. `predict_env` doesn't pin `pyarrow`), then install this repo as an
editable package with the `dev` extras:

```bash
conda activate data_prep2   # or predict_env
pip install -e ".[dev]"
```

```bash
pytest                # run the test suite
ruff format .          # apply formatting
ruff check .           # lint
```

## Running the pipeline

Every script can be run directly with `python`, or through its `run-*.sh`
wrapper, which activates the right conda env via `conda run` and is what the
DPS registration YAMLs (`register_*.yml`) invoke in production.

### 1. Find fire-coincident ATL08 tracks *(optional)*

```bash
python coincident_fire_atl08.py \
  --tile_num 3364 \
  --fire_path fires.gpkg \
  --atl08_years 2019 2020 2021 \
  --atl08_paths s3://bucket/atl08_2019.parquet s3://bucket/atl08_2020.parquet s3://bucket/atl08_2021.parquet \
  --hls_path s3://bucket/hls_2019.tif
```

```bash
# wrapper -- note the quoting: run-coincident_fire_atl08.sh passes $3/$4
# unquoted, so a single quoted, space-separated argument gets word-split
# into the multiple values argparse's nargs='+' expects.
./run-coincident_fire_atl08.sh 3364 fires.gpkg \
  "2019 2020 2021" \
  "s3://bucket/atl08_2019.parquet s3://bucket/atl08_2020.parquet s3://bucket/atl08_2021.parquet" \
  s3://bucket/hls_2019.tif
```

### 2. Extract training-data TFRecords

```bash
python data_prep.py \
  --tile_num 3364 \
  --hls_tindex hls_tindex.csv \
  --atl08_tindex atl08_tindex.csv \
  --agb \
  --out_dir output/
```

`--topo_tindex` defaults to a shared MAAP S3 tindex, so it can usually be
omitted. `--rh` (default `h_canopy`), `--patch_size` (128), `--overlap`
(32), and `--ndval_thresh` (0.30) are also overridable.

```bash
# wrapper
./run-data-prep.sh \
  --tile_num 3364 \
  --hls_tindex hls_tindex.csv \
  --atl08_tindex atl08_tindex.csv \
  --agb
```

### 3. Predict a single HLS scene

```bash
python predict.py \
  --hls_path hls_2023.tif \
  --topo_path topo.tif \
  --lc_path landcover.tif \
  --model_path model.keras \
  --out_raster_path pred_agb_2023.tif \
  --agb
```

```bash
# wrapper (positional: hls_path topo_path lc_path model_path out_raster_path
# patch_size step_size ndval batch_size)
./run-predict.sh hls_2023.tif topo.tif landcover.tif model.keras \
  pred_agb_2023.tif 128 100 -9999 64
```

### 4. Predict across every available year for a tile

This is the algorithm actually registered on DPS
(`register_predict_all_years.yml`).

```bash
python predict_all_years.py \
  --tile_num 3364 \
  --hls_tindex hls_tindex.csv \
  --topo_path topo.tif \
  --lc_path landcover.tif \
  --model_path model.keras \
  --agb
```

```bash
# wrapper
./run-predict-all-years.sh \
  --tile_num 3364 \
  --hls_tindex hls_tindex.csv \
  --topo_path topo.tif \
  --lc_path landcover.tif \
  --model_path model.keras \
  --agb
```

## DPS deployment

`register_predict_all_years.yml` and `register_coincident_fire_atl08.yml`
register those two algorithms on MAAP's DPS; both build against
`predict_env` via `build-env-predict.sh`. `data_prep.py` is no longer run
through DPS -- it's fast enough to run locally/manually, so there's no
`register_data-prep.yml`.
