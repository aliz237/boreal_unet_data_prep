import argparse
import logging
from pathlib import Path

import numpy as np
import rasterio
import s3fs
import tensorflow as tf
from keras.models import load_model
from osgeo import gdal
from rasterio.windows import Window

from constants import Consts
from raster_utils import align_if_needed, gapfill, normalize_bands
from stac_search import get_single_path, get_year_paths, load_items

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info('Devices: %s', tf.config.list_physical_devices())


def download_to_local(s3_path, local_dir='input'):
    """Downloads an s3:// asset to local_dir and returns the local path (a no-op,
    returning the path unchanged, if it isn't an s3:// URL).

    predict_raster()'s raster I/O (normalize_bands, align_if_needed, rasterio.open)
    needs local filesystem paths -- e.g. normalize_bands writes its output alongside
    the input via `path.replace('.tif', '_norm.tif')`, which isn't meaningful for an
    s3:// href. STAC-resolved asset hrefs are always s3:// URLs, so every caller must
    materialize them locally first.

    This is an explicit step here rather than relying on DPS's automatic single-file
    "file input" staging (as topo/land-cover paths used to, before the STAC
    migration): that only works cleanly for one fixed file per input. HLS needs a
    different file per year, which can't be expressed as a fixed DPS file input
    without editing the registration YAML every time a new year is processed -- the
    same reason predict_all_years.py already downloaded HLS this way before topo/
    land-cover moved onto the STAC catalog too.
    """
    if not str(s3_path).startswith('s3://'):
        return str(s3_path)
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    local_path = str(Path(local_dir) / Path(s3_path).name)
    s3 = s3fs.S3FileSystem(anon=False, client_kwargs={'region_name': 'us-west-2'})
    logger.info('Downloading %s', s3_path)
    s3.get_file(s3_path, local_path)
    return local_path


def predict_raster(
    hls_path,
    topo_path,
    lc_path,
    out_raster_path,
    model_path,
    patch_size=128,
    step_size=100,
    ndval=-9999,
    batch_size=64,
    agb=False,
    max_na_block=3,
    nodata_thresh=0.05,
):
    batch = []
    ulxy = []
    hls_path = normalize_bands(
        hls_path,
        hls_path.replace('.tif', '_norm.tif'),
        Consts.HLS_BANDS,
        ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'nbr'],
    )
    topo_path = normalize_bands(
        topo_path,
        topo_path.replace('.tif', '_norm.tif'),
        Consts.TOPO_BANDS,
        ['slope', 'tsri'],
    )

    hls_path, topo_path, lc_path = align_if_needed(hls_path, topo_path, lc_path)
    topo = rasterio.open(topo_path)

    hls_patches_dropped = 0
    topo_patches_dropped = 0
    ax = np.clip(
        np.minimum(np.linspace(0, 1, patch_size), np.linspace(1, 0, patch_size)) * 5,
        0.01,
        1,
    )
    kernel = np.outer(ax, ax)
    model = load_model(model_path, compile=False)
    scalar = Consts.MAX_AGB if agb else Consts.MAX_HEIGHT
    logger.info('agb:%s, scalar:%s', agb, scalar)
    with rasterio.open(hls_path) as hls:
        w, h = hls.width, hls.height
        out_arr = np.full((h, w), 0, dtype=np.float32)
        count_arr = np.full((h, w), 0, dtype=np.float32)
        meta = hls.meta.copy()

        for j in list(range(0, w, step_size))[:-1] + [w - patch_size]:
            for i in list(range(0, h, step_size))[:-1] + [h - patch_size]:
                win = Window(j, i, patch_size, patch_size)
                hls_arr = hls.read(window=win).astype(np.float32)

                # can't have a null patch
                hls_arr[hls_arr == ndval] = np.nan
                if np.any(np.isnan(hls_arr).all(axis=(1, 2))):
                    continue

                # fill NA
                if not gapfill(hls_arr, max_na_block, nodata_thresh):
                    hls_patches_dropped += 1
                    continue
                # same for topo
                topo_arr = topo.read(window=win).astype(np.float32)
                topo_arr[topo_arr == ndval] = np.nan
                if np.any(np.isnan(topo_arr).all(axis=(1, 2))):
                    continue
                if not gapfill(topo_arr, max_na_block, nodata_thresh):
                    topo_patches_dropped += 1
                    continue

                X = np.concatenate([hls_arr, topo_arr])
                batch.append(np.moveaxis(X, 0, -1))
                ulxy.append((j, i))

                if len(batch) > batch_size:
                    preds = model.predict(np.array(batch), verbose=0) * scalar
                    for (x, y), pred in zip(ulxy, preds):
                        out_arr[y : y + patch_size, x : x + patch_size] += (
                            pred[:, :, 0] * kernel
                        )
                        count_arr[y : y + patch_size, x : x + patch_size] += kernel
                    batch = []
                    ulxy = []

    if batch:
        preds = model.predict(np.array(batch)) * scalar
        for (x, y), pred in zip(ulxy, preds):
            out_arr[y : y + patch_size, x : x + patch_size] += pred[:, :, 0] * kernel
            count_arr[y : y + patch_size, x : x + patch_size] += kernel
        batch = []
        ulxy = []
    # out_arr[count_arr == 0] = ndval
    # out_arr[count_arr != 0] /= count_arr
    out_arr = np.divide(out_arr, count_arr, where=count_arr != 0)
    out_arr[count_arr == 0] = ndval
    # land cover mask
    with rasterio.open(lc_path) as lc:
        lc_arr = lc.read(1)
        out_arr[np.isin(lc_arr, [0, 50, 60, 70, 80, 200])] = ndval

    logger.info('min=%s, min=%s', np.min(out_arr), np.min(out_arr[out_arr != ndval]))
    logger.info(
        '%s hls_patches_dropped, %s topo_patches_dropped',
        hls_patches_dropped,
        topo_patches_dropped,
    )
    meta.update({'count': 1, 'nodata': ndval, 'dtype': 'float32'})
    tmp_tif = out_raster_path.replace('.tif', '_temp.tif')
    with rasterio.open(tmp_tif, 'w', **meta) as o:
        o.write(out_arr, 1)
        band_name = 'AGB' if agb else 'Ht'
        o.set_band_description(1, band_name)
    # write tmp_tif as cog
    gdal.Translate(
        out_raster_path,
        tmp_tif,
        format='COG',
        noData='-9999',
        creationOptions=[
            'COMPRESS=DEFLATE',
            'OVERVIEW_COUNT=4',
            'RESAMPLING=AVERAGE',
            'OVERVIEWS=IGNORE_EXISTING',
            'NUM_THREADS=ALL_CPUS',
            'BLOCKSIZE=512',
        ],
    )
    topo.close()
    Path(tmp_tif).unlink(missing_ok=True)
    Path(hls_path).unlink(missing_ok=True)
    Path(topo_path).unlink(missing_ok=True)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        description='Predicts a vegetation height raster given a HLS, Slope and unet model'
    )
    parse.add_argument('--tile_num', help='boreal tile number', type=int, required=True)
    parse.add_argument(
        '--year', help='HLS composite year to predict on', type=int, required=True
    )
    parse.add_argument(
        '--stac_catalog',
        help=(
            'path to the STAC items GeoParquet table (local path or s3://), built '
            'by build_stac_catalog.py'
        ),
        required=True,
    )
    parse.add_argument(
        '--out_raster_path', help='output predicted raster path', required=True
    )
    parse.add_argument(
        '--input_dir',
        help='local dir to download STAC-resolved assets into',
        default='input',
    )
    parse.add_argument('--model_path', help='path to UNet model', required=True)
    parse.add_argument(
        '--patch_size',
        help='patch size, should be the same as what was used when training the model',
        type=int,
        default=128,
    )
    parse.add_argument(
        '--step_size',
        help='step size for sliding the window of size patch_size over the input rasters',
        type=int,
        default=100,
    )
    parse.add_argument('--ndval', help='nodata value', type=int, default=-9999)
    parse.add_argument(
        '--batch_size',
        help='batch size of image patches passed to model.predict',
        type=int,
        default=64,
    )
    parse.add_argument(
        '--agb',
        help='if true predict agb, o.w predict canopy height',
        action='store_true',
    )

    args = parse.parse_args()
    logger.info(args)

    items = load_items(args.stac_catalog)
    hls_path = get_year_paths(items, Consts.HLS_COLLECTION, args.tile_num)[args.year]
    topo_path = get_single_path(items, Consts.TOPO_COLLECTION, args.tile_num)
    lc_path = get_single_path(items, Consts.LC_COLLECTION, args.tile_num)

    hls_path = download_to_local(hls_path, args.input_dir)
    topo_path = download_to_local(topo_path, args.input_dir)
    lc_path = download_to_local(lc_path, args.input_dir)

    predict_raster(
        hls_path=hls_path,
        topo_path=topo_path,
        lc_path=lc_path,
        out_raster_path=args.out_raster_path,
        model_path=args.model_path,
        patch_size=args.patch_size,
        step_size=args.step_size,
        ndval=args.ndval,
        batch_size=args.batch_size,
        agb=args.agb,
    )
