import logging
from pathlib import Path

import numpy as np
import rasterio
from osgeo import gdal

logger = logging.getLogger(__name__)


def is_lidar_heavy(atl08_arr, min_valid_lidar_per_batch, nodata=-9999):
    # filtering out the low quality atl08 patches
    return np.sum(atl08_arr != nodata) > min_valid_lidar_per_batch


def gapfill(arr, max_na_block=3, nodata_thresh=0.05):
    patch_size = arr.shape[1]
    # divides patch sizes 128, 64, 32 and is large enough
    # 2 or 4 are too small
    filter_size = 8
    bands = list(range(arr.shape[0]))  # fill all bands
    # if over 5% of the Blue band is NaN, drop the patch
    ndfrac = np.isnan(arr[0]).sum() / arr[0].size
    if ndfrac > nodata_thresh:
        logger.info('Gapfill: Dropping low quality patch, ndfrac: %s', ndfrac)
        return False

    for band in bands:
        na_blocks_band = 0
        patch_median = np.nanmedian(arr[band])
        # first try block filling
        for j in range(0, patch_size, filter_size):
            for i in range(0, patch_size, filter_size):
                win = arr[band, i : i + filter_size, j : j + filter_size]
                fill_val = np.nanmedian(win)
                if not np.isnan(fill_val):
                    win[np.isnan(win)] = fill_val
                else:
                    na_blocks_band += 1

        # fill the rest with patch-wide median
        na_mask = np.isnan(arr[band])
        if np.any(na_mask):
            arr[band][na_mask] = patch_median
        # threshold too many NA blocks
        if na_blocks_band > max_na_block:
            logger.info(
                'Gapfill: Dropping low quality patch, na_blocks: %s', na_blocks_band
            )
            return False

    return True


def normalize_bands(
    in_raster_path, out_raster_path, band_defs, band_names, mask_path=None
):
    selected = {k: v for k, v in band_defs.items() if k in band_names}
    if mask_path:
        mask = rasterio.open(mask_path).read(1).astype('int32')
    with rasterio.open(in_raster_path) as src:
        arr = src.read([v['num'] for v in selected.values()]).astype('float32')
        profile = src.profile
    for i, (name, meta) in enumerate(selected.items()):
        validmask = arr[i] != -9999
        if mask_path:
            validmask &= mask == 1
        if meta['norm']:
            arr[i][validmask] = meta['norm'](arr[i][validmask])
        # regardless of whether this band has a norm function: everywhere outside
        # validmask (originally nodata, or masked out by mask_path) must read back
        # as nodata. Without this, a masked-out pixel in a norm=None band (most HLS
        # bands) would keep its raw, non-nodata value -- silently defeating
        # mask_path entirely for those bands.
        arr[i][~validmask] = -9999

    profile.update({'dtype': 'float32', 'count': len(selected), 'nodata': -9999})
    with rasterio.open(out_raster_path, 'w', **profile) as dst:
        dst.write(arr)
    return out_raster_path


def raster_bounds(ds):
    """Returns (xmin, ymin, xmax, ymax) for an open gdal.Dataset."""
    gt = ds.GetGeoTransform()
    xmin, ymax = gt[0], gt[3]
    xmax = xmin + (ds.RasterXSize * gt[1])
    ymin = ymax + (ds.RasterYSize * gt[5])
    return (xmin, ymin, xmax, ymax)


def open_raster_bounds(raster_path):
    """Opens raster_path and returns its (xmin, ymin, xmax, ymax) bounds."""
    with rasterio.open(raster_path) as src:
        b = src.bounds
        return (b.left, b.bottom, b.right, b.top)


def align_if_needed(hls_path, topo_path, lc_path):
    ds1 = gdal.Open(hls_path)
    ds2 = gdal.Open(topo_path)
    ds3 = gdal.Open(lc_path)
    dims = [
        ds1.RasterXSize,
        ds2.RasterXSize,
        ds3.RasterXSize,
        ds1.RasterYSize,
        ds2.RasterYSize,
        ds3.RasterYSize,
    ]
    unique_dims = len(list(set(dims)))
    if unique_dims != 1:
        ext1 = raster_bounds(ds1)
        ext2 = raster_bounds(ds2)
        ext3 = raster_bounds(ds3)
        logger.info(
            'HLS, TOPO, and LC dims dont match: %s, resampling to common grid.', dims
        )

        intersection = [
            max(ext1[0], ext2[0], ext3[0]),
            max(ext1[1], ext2[1], ext3[1]),
            min(ext1[2], ext2[2], ext3[2]),
            min(ext1[3], ext2[3], ext3[3]),
        ]

        options_dict = {
            'outputBounds': intersection,
            'width': 3000,
            'height': 3000,
            'format': 'GTiff',
        }

        hls_path = hls_path.replace('.tif', '_resamp.tif')
        topo_path = topo_path.replace('.tif', '_resamp.tif')
        lc_path = lc_path.replace('.tif', '_resamp.tif')

        gdal.Warp(
            hls_path,
            ds1,
            resampleAlg='bilinear',
            srcNodata=-9999,
            dstNodata=-9999,
            **options_dict,
        )
        gdal.Warp(
            topo_path,
            ds2,
            resampleAlg='bilinear',
            srcNodata=-9999,
            dstNodata=-9999,
            **options_dict,
        )
        gdal.Warp(
            lc_path, ds3, resampleAlg='near', srcNodata=0, dstNodata=0, **options_dict
        )

        ds1 = ds2 = ds3 = None
        logger.info('Calculated Intersection: %s', intersection)

    return Path(hls_path), Path(topo_path), Path(lc_path)
