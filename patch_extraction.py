import logging
from pathlib import Path

import numpy as np
import rasterio
import tensorflow as tf
from rasterio.windows import Window

from constants import Consts
from raster_utils import gapfill, is_lidar_heavy
from tfrecord_utils import serialize_image_patch

logger = logging.getLogger(__name__)


def _read_atl08_patch(atl08_ds, win):
    arr = atl08_ds.read(window=win).astype(np.float32)
    if arr.ndim != 3:  # could be just RH98 lyr or include AGB lyr
        arr = arr[np.newaxis, ...]
    return arr


def _read_valid_hls_patch(hls_ds, win, ndval, max_na):
    """Reads and gap-fills an HLS patch.

    Returns (arr, reason, nd_frac): reason is None if usable, else the failed gate;
    nd_frac is the worst band's pre-gapfill nodata fraction.
    """
    arr = hls_ds.read(window=win).astype(np.float32)
    # can't have nulls in HLS, if >= ndval_thresh % is null for any band, drop
    arr[arr == ndval] = np.nan
    nd = np.isnan(arr).sum(axis=(1, 2))
    nd_frac = float(nd.max()) / float(arr.shape[1] * arr.shape[2])
    if np.any(nd >= max_na):
        logger.debug('sparse HLS covergae, dropping patch')
        return None, 'hls_sparse', nd_frac
    if not gapfill(arr):
        logger.debug('Not gapfilling HLS, dropping patch')
        return None, 'hls_gapfill', nd_frac
    return arr, None, nd_frac


def _read_valid_topo_patch(topo_ds, win, ndval, ndval_thresh, patch_size):
    """Reads and gap-fills a topo patch.

    Returns (arr, reason); reason is None when the patch is usable.
    """
    arr = topo_ds.read(window=win).astype(np.float32)
    arr[arr == ndval] = np.nan
    if np.any(np.isnan(arr).sum(axis=(1, 2)) >= ndval_thresh * patch_size**2):
        logger.debug('sparse TOPO covergae, dropping patch')
        return None, 'topo_sparse'
    if not gapfill(arr):
        return None, 'topo_gapfill'
    return arr, None


def _window_label_stats(a_arr, ndval):
    """ATL08 label counts and AGB stats (Mg/ha) for one window."""
    rh = a_arr[0]
    out = {
        'n_rh': int(np.count_nonzero(rh != ndval)),
        'n_agb': 0,
        'mean_agb': np.nan,
        'p95_agb': np.nan,
        'max_agb': np.nan,
    }
    if a_arr.shape[0] > 1:
        v = a_arr[-1]
        v = v[(v != ndval) & np.isfinite(v)] * Consts.MAX_AGB
        if v.size:
            out.update(
                n_agb=int(v.size),
                mean_agb=float(v.mean()),
                p95_agb=float(np.quantile(v, 0.95)),
                max_agb=float(v.max()),
            )
    return out


def _window_mean_slope(topo_ds, win, ndval):
    """Mean slope in degrees over a window of the normalized topo stack."""
    s = topo_ds.read(1, window=win).astype(np.float32)
    s = s[s != ndval]
    return float(np.nanmean(s)) * Consts.MAX_SLOPE if s.size else np.nan


def _write_drop_log(rows, drop_log_path):
    """Write the per-window drop log and log a by-reason summary.

    agb_label_weighted is the label-count-weighted mean AGB per reason.
    """
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        logger.info('drop log empty, nothing written')
        return
    drop_log_path = Path(drop_log_path)
    drop_log_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(drop_log_path, index=False)

    g = df.groupby('reason').agg(
        n_windows=('reason', 'size'),
        n_labels=('n_agb', 'sum'),
        mean_slope=('mean_slope', 'mean'),
        mean_nd_frac=('nd_frac', 'mean'),
    )
    w = df[df['n_agb'] > 0].copy()
    w['_s'] = w['mean_agb'] * w['n_agb']
    by_reason = w.groupby('reason')
    g['agb_label_weighted'] = by_reason['_s'].sum() / by_reason['n_agb'].sum()
    logger.info('wrote %s\n%s', drop_log_path, g.to_string())


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

                    h1_arr, _, _ = _read_valid_hls_patch(h1, win, ndval, max_na)
                    h2_arr, _, _ = _read_valid_hls_patch(h2, win, ndval, max_na)
                    if h1_arr is None or h2_arr is None:
                        continue

                    tp_arr, _ = _read_valid_topo_patch(
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
    drop_log_path=None,  # csv of per-window drop reasons + label stats; None disables
):
    """Like extract_patches_tfrec, but writes one (H, W, D) record per year
    instead of stacking consecutive years into a (2, H, W, D) before/after pair.

    drop_log_path logs every window's keep/drop reason and label stats, to check
    whether a filter biases the training labels. Diagnostic only (slower).
    """
    years, step_size, min_n, max_na, tfw = _init_extraction(
        hls_paths, tfrecord_path, patch_size, overlap, ndval_thresh
    )
    n = 0
    all_dims = set()
    drop_log = [] if drop_log_path else None
    tile = Path(tfrecord_path).name.split('_')[0]
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

                    row = None
                    if drop_log is not None:
                        # read slope before the lidar gate so its drops carry it too
                        row = {
                            'tile': tile,
                            'year': year,
                            'j': j,
                            'i': i,
                            'nd_frac': np.nan,
                            'mean_slope': _window_mean_slope(tp, win, ndval),
                        }
                        row.update(_window_label_stats(a_arr, ndval))

                    # look for an ~diagonal lidar track across the patch
                    if not is_lidar_heavy(a_arr[0], min_n):
                        logger.debug('sparse lidar covergae, dropping patch')
                        if row is not None:
                            row['reason'] = 'lidar_sparse'
                            drop_log.append(row)
                        continue

                    h_arr, reason, nd_frac = _read_valid_hls_patch(h, win, ndval, max_na)
                    if row is not None:
                        row['nd_frac'] = nd_frac
                    if h_arr is None:
                        if row is not None:
                            row['reason'] = reason
                            drop_log.append(row)
                        continue

                    tp_arr, reason = _read_valid_topo_patch(
                        tp, win, ndval, ndval_thresh, patch_size
                    )
                    if tp_arr is None:
                        if row is not None:
                            row['reason'] = reason
                            drop_log.append(row)
                        continue

                    if row is not None:
                        row['reason'] = 'kept'
                        drop_log.append(row)

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
    if drop_log is not None:
        _write_drop_log(drop_log, drop_log_path)
    _finalize_tfrecord(tfrecord_path, n, all_dims, hls_paths[years[-1]])
