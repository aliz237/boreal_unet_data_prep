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


def _label_count_group(reason, n_agb, min_n):
    """Bucket lidar_sparse drops by label count; other reasons pass through."""
    if reason != 'lidar_sparse':
        return reason
    if n_agb == 0:
        return 'lidar_sparse:0'
    lo = max(1, min_n // 3)
    return (
        f'lidar_sparse:1-{lo - 1}' if n_agb < lo
        else f'lidar_sparse:{lo}-{min_n}'
    )


def _drop_log_summary(df, min_n):
    """Summary of a per-window drop log, grouped by gate and label count.

    agb_label_weighted is the mean AGB weighted by label count, not window count.
    p_fail_* is the share of each group's windows that also fail each gate.
    pct_labels is the group's share of all ATL08 labels in the tile.
    """
    df = df.copy()
    df['group'] = [
        _label_count_group(r, n, min_n) for r, n in zip(df['reason'], df['n_agb'])
    ]
    g = df.groupby('group').agg(
        n_windows=('group', 'size'),
        n_labels=('n_agb', 'sum'),
        mean_slope=('mean_slope', 'mean'),
        mean_nd_frac=('nd_frac', 'mean'),
        p_fail_lidar=('fail_lidar', 'mean'),
        p_fail_hls=('fail_hls', 'mean'),
        p_fail_topo=('fail_topo', 'mean'),
    )
    total = g['n_labels'].sum()
    g['pct_labels'] = g['n_labels'] / total if total else np.nan

    w = df[df['n_agb'] > 0].copy()
    w['_s'] = w['mean_agb'] * w['n_agb']
    by_group = w.groupby('group')
    g['agb_label_weighted'] = by_group['_s'].sum() / by_group['n_agb'].sum()

    g = g.reset_index()
    # kept first, then whatever holds the most labels
    g['_k'] = (g['group'] != 'kept').astype(int)
    return (
        g.sort_values(['_k', 'n_labels'], ascending=[True, False])
        .drop(columns='_k')
        .reset_index(drop=True)
    )


_WINDOW_WEIGHTED = [
    'mean_slope',
    'mean_nd_frac',
    'p_fail_lidar',
    'p_fail_hls',
    'p_fail_topo',
]


def combine_drop_summaries(summary_paths):
    """Pool per-tile drop-log summaries, re-weighting by window or label count."""
    import pandas as pd

    df = pd.concat(
        [
            pd.read_csv(f).assign(tile=Path(f).name.split('_')[0])
            for f in summary_paths
        ],
        ignore_index=True,
    )
    for c in _WINDOW_WEIGHTED:
        df[f'_{c}'] = df[c] * df['n_windows']
    df['_agb'] = df['agb_label_weighted'] * df['n_labels']

    g = df.groupby('group').agg(
        n_tiles=('tile', 'nunique'),
        n_windows=('n_windows', 'sum'),
        n_labels=('n_labels', 'sum'),
        _agb=('_agb', 'sum'),
        **{f'_{c}': (f'_{c}', 'sum') for c in _WINDOW_WEIGHTED},
    )
    for c in _WINDOW_WEIGHTED:
        g[c] = g[f'_{c}'] / g['n_windows']
    g['agb_label_weighted'] = g['_agb'] / g['n_labels'].replace(0, np.nan)
    g['pct_labels'] = g['n_labels'] / g['n_labels'].sum()

    g = g[
        ['n_tiles', 'n_windows', 'n_labels', 'pct_labels']
        + _WINDOW_WEIGHTED
        + ['agb_label_weighted']
    ].reset_index()
    g['_k'] = (g['group'] != 'kept').astype(int)
    return (
        g.sort_values(['_k', 'n_labels'], ascending=[True, False])
        .drop(columns='_k')
        .reset_index(drop=True)
    )


def gate_footprint(window_log_paths):
    """Labels and mean AGB rejected by each gate independently, pooled over tiles.

    Summary groups only blame the first failing gate, which hides later gates.
    """
    import pandas as pd

    df = pd.concat([pd.read_csv(f) for f in window_log_paths], ignore_index=True)
    df = df[df['n_agb'] > 0]
    out = []
    for gate in ('fail_lidar', 'fail_hls', 'fail_topo'):
        for failed in (False, True):
            sub = df[df[gate] == failed]
            n = sub['n_agb'].sum()
            out.append({
                'gate': gate,
                'rejected': failed,
                'n_windows': len(sub),
                'n_labels': int(n),
                'pct_labels': n / df['n_agb'].sum() if len(df) else np.nan,
                'agb_label_weighted': (
                    (sub['mean_agb'] * sub['n_agb']).sum() / n if n else np.nan
                ),
            })
    return pd.DataFrame(out)


def _write_drop_log(rows, drop_log_path, min_n, write_windows=True):
    """Write the by-reason summary, and optionally the per-window rows."""
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        logger.info('drop log empty, nothing written')
        return None
    drop_log_path = Path(drop_log_path)
    drop_log_path.parent.mkdir(parents=True, exist_ok=True)

    summary = _drop_log_summary(df, min_n)
    summary_path = drop_log_path.with_name(drop_log_path.stem + '_summary.csv')
    summary.to_csv(summary_path, index=False)
    if write_windows:
        df.to_csv(drop_log_path, index=False)

    logger.info('wrote %s\n%s', summary_path, summary.to_string(index=False))
    return summary


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
                    # look for an ~diagonal lidar track across the patch
                    lidar_ok = is_lidar_heavy(a_arr[0], min_n)

                    # with the drop log on, every gate runs on every window
                    logging_all = drop_log is not None
                    if not lidar_ok and not logging_all:
                        logger.debug('sparse lidar covergae, dropping patch')
                        continue

                    h_arr, hls_reason, nd_frac = _read_valid_hls_patch(
                        h, win, ndval, max_na
                    )
                    if h_arr is None and not logging_all:
                        continue

                    tp_arr, topo_reason = _read_valid_topo_patch(
                        tp, win, ndval, ndval_thresh, patch_size
                    )
                    if tp_arr is None and not logging_all:
                        continue

                    if logging_all:
                        reason = (
                            'lidar_sparse'
                            if not lidar_ok
                            else hls_reason or topo_reason or 'kept'
                        )
                        drop_log.append(
                            {
                                'tile': tile,
                                'year': year,
                                'j': j,
                                'i': i,
                                'reason': reason,
                                'fail_lidar': not lidar_ok,
                                'fail_hls': hls_reason is not None,
                                'fail_topo': topo_reason is not None,
                                'nd_frac': nd_frac,
                                'mean_slope': _window_mean_slope(tp, win, ndval),
                                **_window_label_stats(a_arr, ndval),
                            }
                        )
                        if reason != 'kept':
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
    if drop_log is not None:
        _write_drop_log(drop_log, drop_log_path, min_n)
    _finalize_tfrecord(tfrecord_path, n, all_dims, hls_paths[years[-1]])
