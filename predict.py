import subprocess
from pathlib import Path
import argparse
import logging

import numpy as np
from osgeo import gdal
import rasterio
from rasterio.windows import Window

import tensorflow as tf
from keras.models import load_model

from data_prep import gapfill, align_if_needed, normalize_bands, Consts
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.info(f"Devices: {tf.config.list_physical_devices()}")


def masked_mse_loss(mask_value=-9999):
    def loss(y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, mask_value), tf.float32)
        squared_error = tf.square(y_true - y_pred) * mask
        return tf.reduce_sum(squared_error) / (tf.reduce_sum(mask) + 1e-6)
    return loss

def masked_mae_loss(mask_value=-9999):
    def loss(y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, mask_value), tf.float32)
        abs_error = tf.abs(y_true - y_pred) * mask
        return tf.reduce_sum(abs_error) / (tf.reduce_sum(mask) + 1e-6)
    return loss

def predict_raster(hls_path, topo_path, lc_path, out_raster_path, model_path,
                   patch_size=128, step_size=100, ndval=-9999, batch_size=64, agb=False):
    batch = []
    ulxy = []
    hls_path = normalize_bands(
        hls_path,
        hls_path.replace('.tif', '_norm.tif'),
        Consts.HLS_BANDS,
        ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'nbr']
    )
    topo_path = normalize_bands(
        topo_path,
        topo_path.replace('.tif', '_norm.tif'),
        Consts.TOPO_BANDS,
        ['slope', 'tsri']
    )

    hls_path, topo_path, lc_path = align_if_needed(hls_path, topo_path, lc_path)
    topo = rasterio.open(topo_path)

    hls_patches_dropped = 0
    topo_patches_dropped = 0    
    ax = np.clip(np.minimum(np.linspace(0, 1, patch_size), np.linspace(1, 0, patch_size)) * 5, 0.01, 1)
    kernel = np.outer(ax, ax)
    model = load_model(model_path, custom_objects={'loss': masked_mae_loss})
    scalar = Consts.MAX_AGB if agb else Consts.MAX_HEIGHT
    logger.info(f'agb:{agb}, scalar:{scalar}')
    with rasterio.open(hls_path) as hls:
        w, h, c = hls.width, hls.height, hls.count
        out_arr = np.full((h, w), 0, dtype=np.float32)
        count_arr = np.full((h, w), 0, dtype=np.float32)
        meta = hls.meta.copy()

        for j in list(range(0, w, step_size))[:-1] + [w-patch_size]:
            for i in list(range(0, h, step_size))[:-1] + [h-patch_size]:

                win = Window(j, i, patch_size, patch_size)
                hls_arr = hls.read(window=win).astype(np.float32)

                # can't have a null patch
                hls_arr[hls_arr == ndval] = np.nan
                if np.any(np.isnan(hls_arr).all(axis=(1,2))):
                    continue
                
                # fill NA
                if not gapfill(hls_arr):
                    hls_patches_dropped += 1
                    continue
                # same for topo
                topo_arr = topo.read(window=win).astype(np.float32)
                topo_arr[topo_arr == ndval] = np.nan
                if np.any(np.isnan(topo_arr).all(axis=(1,2))):
                    continue
                if not gapfill(topo_arr):
                    topo_patches_dropped += 1
                    continue

                X = np.concatenate([hls_arr, topo_arr])
                batch.append(np.moveaxis(X, 0, -1))
                ulxy.append((j, i))

                if len(batch) > batch_size:
                    preds = model.predict(np.array(batch)) * scalar
                    for (x, y), pred in zip(ulxy, preds):
                        out_arr[y:y+patch_size, x:x+patch_size] += pred[:,:,0] * kernel
                        count_arr[y:y+patch_size, x:x+patch_size] += kernel
                    batch = []
                    ulxy = []
    
    if batch:
        preds = model.predict(np.array(batch)) * scalar
        for (x, y), pred in zip(ulxy, preds):
            out_arr[y:y+patch_size, x:x+patch_size] += pred[:,:,0] * kernel
            count_arr[y:y+patch_size, x:x+patch_size] += kernel
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

    logger.info(f'min={np.min(out_arr)}, min={np.min(out_arr[out_arr != ndval])}')
    logger.info(f"""{hls_patches_dropped} hls_patches_dropped,
    {topo_patches_dropped} topo_patches_dropped"""
    )
    meta.update({'count': 1, 'nodata': ndval, 'dtype': 'float32'})
    tmp_tif = out_raster_path.replace('.tif', '_temp.tif')
    with rasterio.open(tmp_tif, 'w', **meta) as o:
        o.write(out_arr, 1)
        o.set_band_description(1, 'Ht')
    # write tmp_tif as cog
    gdal.Translate(
        out_raster_path,
        tmp_tif,
        format="COG",
        noData='-9999',
        creationOptions=[
            "COMPRESS=DEFLATE",
            "OVERVIEW_COUNT=4",
            "RESAMPLING=AVERAGE",
            "OVERVIEWS=IGNORE_EXISTING",
            'NUM_THREADS=ALL_CPUS',
            "BLOCKSIZE=512"
        ]
    )
    topo.close()
    Path(tmp_tif).unlink(missing_ok=True)
    Path(hls_path).unlink(missing_ok=True)
    Path(topo_path).unlink(missing_ok=True)


if __name__ == "__main__":
    parse = argparse.ArgumentParser(
        description="Predicts a vegetation height raster given a HLS, Slope and unet model"
    )
    parse.add_argument("--hls_path", help="HLS image path", required=True)
    parse.add_argument(
        "--topo_path", help="topo image path with slope as second band", required=True
    )
    parse.add_argument("--lc_path", help="land cover image path", required=True)
    parse.add_argument("--out_raster_path", help="output predicted raster path", required=True)
    parse.add_argument("--model_path", help="path to UNet model", required=True)
    parse.add_argument("--patch_size", help="patch size, should be the same as what was used when training the model", type=int, default=128)
    parse.add_argument("--step_size", help="step size for sliding the window of size patch_size over the input rasters", type=int, default=100)
    parse.add_argument("--ndval", help="nodata value", type=int, default=-9999)
    parse.add_argument("--batch_size", help="batch size of image patches passed to model.predict", type=int, default=64)
    parse.add_argument("--agb", help="if true predict agb, o.w predict canopy height", action='store_true')

    args = parse.parse_args()
    logger.info(args)
    predict_raster(**vars(args))
