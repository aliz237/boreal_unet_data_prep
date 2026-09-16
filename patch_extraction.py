import logging

import numpy as np
import rasterio
import tensorflow as tf
from rasterio.windows import Window

from raster_utils import gapfill, is_lidar_heavy
from tfrecord_utils import serialize_image_patch

logger = logging.getLogger(__name__)


def _read_atl08_patch(atl08_ds, win):
    arr = atl08_ds.read(window=win).astype(np.float32)
    if arr.ndim != 3:  # could be just RH98 lyr or include AGB lyr
        arr = arr[np.newaxis, ...]
    return arr


def _read_valid_hls_patch(hls_ds, win, ndval, max_na):
    """Reads and gap-fills an HLS patch, or returns None if it's too sparse."""
    arr = hls_ds.read(window=win).astype(np.float32)
    # can't have nulls in HLS, if >= ndval_thresh % is null for any band, drop
    arr[arr == ndval] = np.nan
    if np.any(np.isnan(arr).sum(axis=(1, 2)) >= max_na):
        logger.info('sparse HLS covergae, dropping patch')
        return None
    if not gapfill(arr):
        logger.info('Not gapfilling HLS, dropping patch')
        return None
    return arr


def _read_valid_topo_patch(topo_ds, win, ndval, ndval_thresh, patch_size):
    """Reads and gap-fills a topo patch, or returns None if it's too sparse."""
    arr = topo_ds.read(window=win).astype(np.float32)
    arr[arr == ndval] = np.nan
    if np.any(np.isnan(arr).sum(axis=(1, 2)) >= ndval_thresh * patch_size**2):
        logger.info('sparse TOPO covergae, dropping patch')
        return None
    if not gapfill(arr):
        return None
    return arr


def _init_extraction(hls_paths, tfrecord_path, patch_size, overlap, ndval_thresh):
    """Shared setup for both extraction loops: sorted years, sliding-window step
    size, lidar/nodata thresholds, and the open TFRecordWriter."""
    years = sorted(hls_paths.keys())
    step_size = patch_size - overlap
    # 120 is median valid pixel count of lidar track in ATL08 128x128 patches
    # the other one is 70% of diagonal of a patch (so close to complete and decent lidar track)
    min_n = int(min(patch_size * np.sqrt(2) * 0.7, 120))
    max_na = ndval_thresh * patch_size**2
    tfw = tf.io.TFRecordWriter(
        str(tfrecord_path), options=tf.io.TFRecordOptions(compression_type='GZIP')
    )
    return years, step_size, min_n, max_na, tfw


def _finalize_tfrecord(tfrecord_path, n, all_dims, sample_hls_path):
    if len(all_dims) != 1:
        logger.info('shape mismatch ...')
    logger.info('Shapes: %s', all_dims)

    if n == 0:
        logger.info('No patches extracted from %s!', sample_hls_path)
        tfrecord_path.unlink(missing_ok=True)
    else:
        # rename the tfrecord file to include the record count
        tfrecord_path.rename(
            tfrecord_path.with_name(
                tfrecord_path.name.replace('.tfrecord', f'_{n}.tfrecord')
            )
        )
        logger.info('%s records saved', n)


def extract_patches_tfrec(
    hls_paths,  # dict of {year: path} to all hls years
    atl08_paths,  # dict of {year: path} to all atl08 years
    topo_path,  # there's only one topo path
    tfrecord_path,  # output tfrecord path
    patch_size=128,
    ndval=-9999,
    overlap=32,
    ndval_thresh=0.30,
    fire_years=None,  # set of years with a fire mask baked into hls_paths, or None
):
    years, step_size, min_n, max_na, tfw = _init_extraction(
        hls_paths, tfrecord_path, patch_size, overlap, ndval_thresh
    )
    n = 0
    all_dims = set()
    for t1, t2 in zip(years[:-1], years[1:]):
        if fire_years is not None and t1 not in fire_years and t2 not in fire_years:
            # fire-augmentation mode: a pair only has a chance of producing patches
            # if at least one of its years has fire coverage
            logger.info('neither %s nor %s has fire coverage, skipping pair', t1, t2)
            continue
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
            for j in range(0, h1.width - patch_size + 1, step_size):
                for i in range(0, h1.height - patch_size + 1, step_size):
                    # (j, i) is the top-left corner of patch
                    win = Window(j, i, patch_size, patch_size)

                    a1_arr = _read_atl08_patch(a1, win)
                    a2_arr = _read_atl08_patch(a2, win)
                    # look for an ~diagonal lidar track across the patch
                    if not (
                        is_lidar_heavy(a1_arr[0], min_n)
                        or is_lidar_heavy(a2_arr[0], min_n)
                    ):
                        logger.debug('sparse lidar covergae, dropping patch')
                        continue

                    h1_arr = _read_valid_hls_patch(h1, win, ndval, max_na)
                    h2_arr = _read_valid_hls_patch(h2, win, ndval, max_na)
                    if h1_arr is None or h2_arr is None:
                        continue

                    tp_arr = _read_valid_topo_patch(
                        tp, win, ndval, ndval_thresh, patch_size
                    )
                    if tp_arr is None:
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
    _finalize_tfrecord(tfrecord_path, n, all_dims, hls_paths[years[-1]])


def extract_patches_tfrec_per_year(
    hls_paths,  # dict of {year: path} to all hls years
    atl08_paths,  # dict of {year: path} to all atl08 years
    topo_path,  # there's only one topo path
    tfrecord_path,  # output tfrecord path
    patch_size=128,
    ndval=-9999,
    overlap=32,
    ndval_thresh=0.30,
    fire_years=None,  # set of years with a fire mask baked into hls_paths, or None
):
    """Like extract_patches_tfrec, but writes one (H, W, D) record per year
    instead of stacking consecutive years into a (2, H, W, D) before/after pair.
    """
    years, step_size, min_n, max_na, tfw = _init_extraction(
        hls_paths, tfrecord_path, patch_size, overlap, ndval_thresh
    )
    n = 0
    all_dims = set()
    for year in years:
        if fire_years is not None and year not in fire_years:
            logger.info('%s has no fire coverage, skipping', year)
            continue
        logger.info('year:%s', year)
        with (
            rasterio.open(hls_paths[year]) as h,
            rasterio.open(atl08_paths[year]) as a,
            rasterio.open(topo_path) as tp,
        ):
            patch_depth = h.count + tp.count + a.count
            for j in range(0, h.width - patch_size + 1, step_size):
                for i in range(0, h.height - patch_size + 1, step_size):
                    # (j, i) is the top-left corner of patch
                    win = Window(j, i, patch_size, patch_size)

                    a_arr = _read_atl08_patch(a, win)
                    # look for an ~diagonal lidar track across the patch
                    if not is_lidar_heavy(a_arr[0], min_n):
                        logger.debug('sparse lidar covergae, dropping patch')
                        continue

                    h_arr = _read_valid_hls_patch(h, win, ndval, max_na)
                    if h_arr is None:
                        continue

                    tp_arr = _read_valid_topo_patch(
                        tp, win, ndval, ndval_thresh, patch_size
                    )
                    if tp_arr is None:
                        continue

                    # save patches on disk
                    n += 1
                    # prep to write as TFrecord
                    # concat hls, topo features and atl08 label to build one training example
                    arr = np.concatenate([h_arr, tp_arr, a_arr])
                    # reorder as needed by model.fit, channels last
                    arr = np.moveaxis(arr, 0, -1)
                    all_dims.add((h_arr.shape, tp_arr.shape, a_arr.shape, arr.shape))

                    ser = serialize_image_patch(arr, patch_size, patch_depth)
                    tfw.write(ser.numpy())

                    if n % 100 == 0:
                        logger.info('wrote %s records', n)

            logger.info('wrote %s records', n)
    tfw.close()
    _finalize_tfrecord(tfrecord_path, n, all_dims, hls_paths[years[-1]])
