import logging

import numpy as np
import rasterio
import tensorflow as tf
from rasterio.windows import Window

from raster_utils import gapfill, is_lidar_heavy
from tfrecord_utils import serialize_image_patch

logger = logging.getLogger(__name__)


def extract_patches_tfrec(
    hls_paths,  # dict of {year: path} to all hls years
    atl08_paths,  # dict of {year: path} to all atl08 years
    topo_path,  # there's only one topo path
    tfrecord_path,  # output tfrecord path
    patch_size=128,
    ndval=-9999,
    overlap=32,
    ndval_thresh=0.30,
):
    years = sorted(list(hls_paths.keys()))
    step_size = patch_size - overlap
    n = 0
    min_n = int(min(patch_size * np.sqrt(2) * 0.7, 120))
    max_na = ndval_thresh * patch_size**2
    tfw = tf.io.TFRecordWriter(
        str(tfrecord_path), options=tf.io.TFRecordOptions(compression_type='GZIP')
    )
    all_dims = set()
    for t1, t2 in zip(years[:-1], years[1:]):
        logger.info('t1:%s, t2:%s', t1, t2)
        with (
            rasterio.open(hls_paths[t1]) as h1,
            rasterio.open(hls_paths[t2]) as h2,
            rasterio.open(atl08_paths[t1]) as a1,
            rasterio.open(atl08_paths[t2]) as a2,
            rasterio.open(topo_path) as tp,
        ):
            patch_depth = h1.count + tp.count + a1.count
            # patch_depth = 11 # 6 HLS spectral channels, 1 NBR, 1 slope, 1 TSRI, and 2 atl08 label.
            # 120 is median valid pixel count of lidar track in ATL08 128x128 patches
            # the other one is 70% of diagonal of a patch (so close to complete and decent lidar track)
            for j in range(0, h1.width - patch_size + 1, step_size):
                for i in range(0, h1.height - patch_size + 1, step_size):
                    # (j, i) is the top-left corner of patch
                    win = Window(j, i, patch_size, patch_size)

                    # read ATL08 patch (1-band)
                    a1_arr = a1.read(window=win).astype(np.float32)
                    a2_arr = a2.read(window=win).astype(np.float32)
                    if a1_arr.ndim != 3:  # could be just RH98 lyr or include AGB lyr
                        a1_arr = a1_arr[np.newaxis, ...]
                        a2_arr = a2_arr[np.newaxis, ...]
                    # look for an ~diagonal lidar track across the patch
                    if not (
                        is_lidar_heavy(a1_arr[0], min_n)
                        or is_lidar_heavy(a2_arr[0], min_n)
                    ):
                        logger.debug('sparse lidar covergae, dropping patch')
                        continue
                    # read corresponding HLS patch
                    h1_arr = h1.read(window=win).astype(np.float32)
                    h2_arr = h2.read(window=win).astype(np.float32)
                    # can't have nulls in HLS, if >= ndval_thresh % is null for any band, continue
                    h1_arr[h1_arr == ndval] = np.nan
                    h2_arr[h2_arr == ndval] = np.nan
                    if np.any(np.isnan(h1_arr).sum(axis=(1, 2)) >= max_na) or np.any(
                        np.isnan(h2_arr).sum(axis=(1, 2)) >= max_na
                    ):
                        logger.info('sparse HLS covergae, dropping patch')
                        continue
                    # fill NA
                    if not (gapfill(h1_arr) and gapfill(h2_arr)):
                        logger.info('Not gapfilling HLS, dropping patch')
                        continue
                    # same for topo
                    tp_arr = tp.read(window=win).astype(np.float32)
                    tp_arr[tp_arr == ndval] = np.nan
                    if np.any(
                        np.isnan(tp_arr).sum(axis=(1, 2)) >= ndval_thresh * patch_size**2
                    ):
                        logger.info('sparse TOPO covergae, dropping patch')
                        continue
                    # fill NA
                    if not gapfill(tp_arr):
                        continue
                    # save patches on disk
                    n += 1
                    # prep to write as TFrecord
                    # concat hls, topo features and atl08 label to build one training example
                    arr1 = np.concatenate([h1_arr, tp_arr, a1_arr])
                    arr2 = np.concatenate([h2_arr, tp_arr, a2_arr])
                    # reorder as needed by model.fit, channels last
                    arr1 = np.moveaxis(arr1, 0, -1)
                    arr2 = np.moveaxis(arr2, 0, -1)
                    arr = np.stack([arr1, arr2], axis=0)
                    all_dims.add(
                        (
                            h1_arr.shape,
                            h2_arr.shape,
                            tp_arr.shape,
                            a1_arr.shape,
                            a2_arr.shape,
                            arr1.shape,
                            arr2.shape,
                            arr.shape,
                        )
                    )

                    ser = serialize_image_patch(arr, patch_size, patch_depth)
                    tfw.write(ser.numpy())

                    if n % 100 == 0:
                        logger.info('wrote %s records', n)

            logger.info('wrote %s records', n)
    tfw.close()
    if len(all_dims) != 1:
        logger.info('shape mismatch ...')

    logger.info('Shapes: %s', all_dims)

    if n == 0:
        logger.info('No patches extracted from %s!', hls_paths[t1])
        tfrecord_path.unlink(missing_ok=True)
    else:
        # rename the tfrecord file to include the record count
        tfrecord_path.rename(
            tfrecord_path.with_name(
                tfrecord_path.name.replace('.tfrecord', f'_{n}.tfrecord')
            )
        )
        logger.info('%s records saved', n)
