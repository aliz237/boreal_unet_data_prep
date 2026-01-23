import subprocess
from pathlib import Path

import numpy as np

import rasterio
from rasterio.windows import Window

import tensorflow as tf
from keras.models import load_model

from data_prep import gapfill, resample_topo_if_needed, subset_HLS_bands

def masked_mse_loss(mask_value=-9999):
    def loss(y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, mask_value), tf.float32)
        squared_error = tf.square(y_true - y_pred) * mask
        return tf.reduce_sum(squared_error) / (tf.reduce_sum(mask) + 1e-6)
    return loss

def predict_raster(hls_path, topo_path, out_raster_path, model_path, patch_size=128, step_size=100, ndval=-9999, batch_size=64):
    batch = []
    ulxy = []
    topo_path = resample_topo_if_needed(Path(topo_path))
    topo = rasterio.open(topo_path)
    hls_path = subset_HLS_bands(Path(hls_path), clean=True)
    hls_patches_dropped = 0
    topo_patches_dropped = 0    
    ax = np.clip(np.minimum(np.linspace(0, 1, patch_size), np.linspace(1, 0, patch_size)) * 5, 0.01, 1)
    kernel = np.outer(ax, ax)
    model = load_model(model_path, custom_objects={'loss': masked_mse_loss})

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
                slope = topo.read([2], window=win).astype(np.float32)
                slope[slope == ndval] = np.nan
                slope = np.clip(slope, 0, 90)
                slope /= 90.0
                if np.isnan(slope).all():
                    continue
                if not gapfill(slope):
                    topo_patches_dropped += 1
                    continue

                X = np.concatenate([hls_arr, slope])
                batch.append(np.moveaxis(X, 0, -1))
                ulxy.append((j, i))

                if len(batch) > batch_size:
                    preds = model.predict(np.array(batch))
                    for (x, y), pred in zip(ulxy, preds):
                        out_arr[y:y+patch_size, x:x+patch_size] += pred[:,:,0] * kernel
                        count_arr[y:y+patch_size, x:x+patch_size] += kernel
                    batch = []
                    ulxy = []
    
    if batch:
        preds = model.predict(np.array(batch))
        for (x, y), pred in zip(ulxy, preds):
            out_arr[y:y+patch_size, x:x+patch_size] += pred[:,:,0] * kernel
            count_arr[y:y+patch_size, x:x+patch_size] += kernel
        batch = []
        ulxy = []
    # out_arr[count_arr == 0] = ndval
    # out_arr[count_arr != 0] /= count_arr
    out_arr = np.divide(out_arr, count_arr, where=count_arr != 0)
    out_arr[count_arr == 0] = ndval
    print(f'min={np.min(out_arr)}, min={np.min(out_arr[out_arr != ndval])}')
    print(f"""{hls_patches_dropped} hls_patches_dropped,
    {topo_patches_dropped} topo_patches_dropped"""
    )
    meta.update({'count': 1, 'nodata': ndval, 'dtype': 'float32'})
    with rasterio.open(out_raster_path, 'w', **meta) as o:
        o.write(out_arr, 1)
        o.set_band_description(1, 'Ht')

    topo.close()

if __name__ == "__main__":
    parse = argparse.ArgumentParser(
        description="Predicts a vegetation height raster given a HLS, Slope and unet model"
    )
    parse.add_argument("--hls_path", help="HLS image path", required=True)
    parse.add_argument(
        "--topo_path", help="topo image path with slope as second band", required=True
    )
    parse.add_argument("--out_raster_path", help="output predicted raster path", required=True)
    parse.add_argument("--model_path", help="path to UNet model", required=True)
    parse.add_argument("--patch_size", help="patch size, should be the same as what was used when training the model", type=int, default=128)
    parse.add_argument("--step_size", help="step size for sliding the window of size patch_size over the input rasters", type=int, default=100)
    parse.add_argument("--ndval", help="nodata value", type=int, default=-9999)
    parse.add_argument("--batch_size", help="batch size of image patches passed to model.predict", type=int, default=64)
    args = parse.parse_args()
    create_training_dataset(**vars(args))
