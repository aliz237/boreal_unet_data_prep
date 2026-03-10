from pathlib import Path
import subprocess
import argparse

import numpy as np
import geopandas as gpd
import pandas as pd

from osgeo import gdal
import rasterio
from rasterio.windows import Window

import tensorflow as tf

class Consts:
    HLS_BANDS = tuple(range(1, 7)) + (12,)
    TOPO_BANDS = (2, 3, 4)
    MAX_HEIGHT = 100.0

# The following functions can be used to convert a value to a type compatible with tf.train.Example.
# stolen from https://www.tensorflow.org/tutorials/load_data/tfrecord
def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

@tf.py_function(Tout=tf.string)
def serialize_image_patch(hls_atl08_arr, patch_size, num_bands):
    arr_ser = tf.io.serialize_tensor(hls_atl08_arr)
    feature = {
        'height': _int64_feature(patch_size),
        'width': _int64_feature(patch_size),
        'depth': _int64_feature(num_bands),
        'arr': _bytes_feature(arr_ser)
    }
    ex = tf.train.Example(features=tf.train.Features(feature=feature))
    return ex.SerializeToString()

def is_lidar_heavy(atl08_arr, patch_size, min_valid_lidar_per_batch, nodata=-9999):
    # filtering out the low quality atl08 patches
    return np.sum(atl08_arr != nodata) > min_valid_lidar_per_batch

def gapfill(arr, nodata_thresh=0.6):
    patch_size = arr.shape[1]
    # divides patch sizes 128, 64, 32 and is large enough 
    # 2 or 4 are too small
    filter_size = 8
    bands = list(range(arr.shape[0])) # fill all bands
    na_blocks = 0
    na_blocks_thresh = int(nodata_thresh * patch_size**2 / filter_size**2)

    for band in bands:
        patch_median = np.nanmedian(arr[band])
        # first try block filling
        for j in range(0, patch_size, filter_size):
            for i in range(0, patch_size, filter_size):
                win = arr[band, i:i+filter_size, j:j+filter_size]
                fill_val = np.nanmedian(win)
                if not np.isnan(fill_val):
                    win[np.isnan(win)] = fill_val
                else:
                    na_blocks += 1

        # fill the rest with patch-wide median
        na_mask = np.isnan(arr[band])
        if np.any(na_mask):
            arr[band][na_mask] = patch_median
        # threshold too many NA blocks
        if na_blocks > na_blocks_thresh:
            return False
    return True

def normalize_stack(in_raster_path, out_raster_path, bands):
    with rasterio.open(in_raster_path) as src:
        arr = src.read(bands).astype('float32')
        profile = src.profile
        for i in range(arr.shape[0]):
            validmask = arr[i] != -9999
            mu = np.mean(arr[i][validmask])
            sd = np.std(arr[i][validmask])
            normalized = (arr[i] - mu) / sd
            arr[i] = np.clip(normalized, -3, 3) / 3.0
            arr[i][~validmask] = -9999

    profile.update({
        'dtype': 'float32',
        'count': len(bands),
        'nodata': -9999
    })

    with rasterio.open(out_raster_path, 'w', **profile) as dst:
        dst.write(arr)

    return out_raster_path


def extract_patches_tfrec(
    hls_path,
    atl08_path,
    topo_path,
    tfrecord_path,
    patch_size=128,
    ndval=-9999,
    overlap=32
):
    hls = rasterio.open(hls_path)
    h = hls.height
    w = hls.width
    c = hls.count

    atl08 = rasterio.open(atl08_path)
    topo = rasterio.open(topo_path)

    assert overlap >= 0 and overlap < patch_size, "invalid overlap!"
    assert patch_size >= 32, "small patch size!"

    tfrecord_path = Path(tfrecord_path)
    step_size = patch_size - overlap
    n = 0
    hls_patches_dropped = 0
    topo_patches_dropped = 0
    ndval_thresh = 0.30
    patch_depth = hls.count + topo.count + atl08.count
    # patch_depth = 11 # 6 HLS spectral channels, 1 NBR, 1 slope, 1 TSRI, 1 TPI, and 1 atl08 label.
    # 120 is median valid pixel count of lidar track in ATL08 128x128 patches
    # the other one is 70% of diagonal of a patch (so close to complete and decent lidar track)
    min_valid_lidar_per_batch = int(min(patch_size * np.sqrt(2) * 0.7, 120))
    tfw = tf.io.TFRecordWriter(
        str(tfrecord_path), options=tf.io.TFRecordOptions(compression_type="GZIP")
    )

    for j in range(0, w - patch_size, step_size):
        for i in range(0, h - patch_size, step_size):
            # (j, i) is the top-left corner of patch
            # read ATL08 patch (1-band), if all null continue
            win = Window(j, i, patch_size, patch_size)
            lab_arr = atl08.read(window=win).astype(np.float32)
            # look for a close to diagonal lidar track across the patch
            if not is_lidar_heavy(lab_arr, patch_size, min_valid_lidar_per_batch):
                continue
            # read corresponding HLS patch
            hls_arr = hls.read(window=win).astype(np.float32)
            # can't have nulls in HLS, if >= ndval_thresh % is null for any band, continue
            hls_arr[hls_arr == ndval] = np.nan
            if np.any(
                np.isnan(hls_arr).sum(axis=(1, 2)) >= ndval_thresh * patch_size**2
            ):
                continue
            # fill NA
            if not gapfill(hls_arr):
                hls_patches_dropped += 1
                continue
            # same for topo
            topo_arr = topo.read(window=win).astype(np.float32)
            topo_arr[topo_arr == ndval] = np.nan
            if np.any(
                np.isnan(topo_arr).sum(axis=(1, 2)) >= ndval_thresh * patch_size**2
            ):
                continue
            # fill NA
            if not gapfill(topo_arr):
                topo_patches_dropped += 1
                continue
            # save patches on disk
            n += 1
            # prep to write as TFrecord
            # concat hls, topo features and atl08 label to build one training example
            arr = np.concatenate([hls_arr, topo_arr, lab_arr])
            # reorder as needed by model.fit, channels last
            arr = np.moveaxis(arr, 0, -1)
            # serialize the arr to write as a tfrecord
            ser = serialize_image_patch(arr, patch_size, patch_depth)
            tfw.write(ser.numpy())
            if n % 100 == 0:
                print(f"wrote {n} records, of total {int(h*w/patch_size**2)}")

    tfw.close()
    hls.close()
    atl08.close()
    topo.close()
    if n == 0:
        print(f"No patches extracted from {hls_path}!")
        tfrecord_path.unlink(missing_ok=True)
    else:
        # rename the tfrecord file to include the record count
        tfrecord_path.rename(
            tfrecord_path.with_name(
                tfrecord_path.name.replace(".tfrecord", f"_{n}.tfrecord")
            )
        )
        print(
            f"""{n} records saved,
              {hls_patches_dropped} hls_patches_dropped,
              {topo_patches_dropped} topo_patches_dropped"""
        )


def atl08_to_raster(atl08_path, hls_path, out_raster_path, ndval=-9999, rh='h_canopy'):
    df = gpd.read_parquet(atl08_path)
    with rasterio.open(hls_path) as hls:
        meta = hls.meta.copy()
        h, w = hls.height, hls.width
        cols, rows = ~hls.transform * (df.geometry.x.values, df.geometry.y.values)
        cols = np.floor(cols).astype(int)
        rows = np.floor(rows).astype(int)
        mask = (rows >= 0) & (cols >= 0) & (rows < hls.height) & (cols < hls.width)
        rows, cols = rows[mask], cols[mask]
        rh98 = df[rh].values[mask]

    out = np.full((h, w), ndval, dtype=np.float32)
    out[rows, cols] = rh98 / Consts.MAX_HEIGHT
    meta.update({"count": 1, "nodata": ndval, "dtype": "float32"})
    with rasterio.open(out_raster_path, "w", **meta) as o:
        o.write(out, 1)


def get_extent(ds):
    gt = ds.GetGeoTransform()
    width = ds.RasterXSize
    height = ds.RasterYSize

    xmin = gt[0]
    ymax = gt[3]
    xmax = xmin + (width * gt[1])
    ymin = ymax + (height * gt[5])

    return [xmin, ymin, xmax, ymax]


def align_if_needed(hls_path, topo_path, lc_path):
    ds1 = gdal.Open(hls_path)
    ds2 = gdal.Open(topo_path)
    ds3 = gdal.Open(lc_path)
    unique_dims = len(list(set([ds1.RasterXSize, ds2.RasterXSize, ds3.RasterXSize,
                 ds1.RasterYSize, ds2.RasterYSize, ds3.RasterYSize])))
    if unique_dims != 1:
        ext1 = get_extent(ds1)
        ext2 = get_extent(ds2)
        ext3 = get_extent(ds3)

        intersection = [
            max(ext1[0], ext2[0], ext3[0]),
            max(ext1[1], ext2[1], ext3[1]),
            min(ext1[2], ext2[2], ext3[2]),
            min(ext1[3], ext2[3], ext3[3])
        ]

        options_dict = {
            'outputBounds':intersection,
            'width':3000,
            'height':3000,
            'format':'GTiff'
        }

        hls_path = hls_path.replace('.tif', '_resamp.tif')
        topo_path = topo_path.replace('.tif', '_resamp.tif')
        lc_path = lc_path.replace('.tif', '_resamp.tif')

        gdal.Warp(hls_path, ds1, resampleAlg='bilinear', srcNodata=-9999, dstNodata=-9999, **options_dict)
        gdal.Warp(topo_path, ds2, resampleAlg='bilinear', srcNodata=-9999, dstNodata=-9999, **options_dict)
        gdal.Warp(lc_path, ds3, resampleAlg='near', srcNodata=0, dstNodata=0, **options_dict)

        ds1 = ds2 = ds3 = None
        print(f"Calculated Intersection: {intersection}")

    return Path(hls_path), Path(topo_path), Path(lc_path)


def create_training_dataset(
    tile_num, year, atl08_path, hls_path, topo_path, patch_size=128, overlap=32, rh='h_canopy'
):

    print("rasterizing atl08 to HLS grid")
    atl08_raster_path = str(Path(atl08_path).with_suffix(".tif"))
    atl08_to_raster(atl08_path, hls_path, atl08_raster_path, rh=rh)

    hls_path = normalize_stack(hls_path, hls_path.replace('.tif', '_norm.tif'), Consts.HLS_BANDS)
    topo_path = normalize_stack(topo_path, topo_path.replace('.tif', '_norm.tif'), Consts.TOPO_BANDS)

    print(f"Extracting patches for tile-year: {tile_num}-{year}")
    extract_patches_tfrec(
        hls_path,
        atl08_raster_path,
        topo_path,
        tfrecord_path=f"output/{tile_num}_{year}_{patch_size}_{overlap}.tfrecord.gz",
        patch_size=patch_size,
        overlap=overlap
    )

if __name__ == "__main__":
    parse = argparse.ArgumentParser(
        description="Creates a tfrecord.gz from patches of HLS, TOPO, and ATL08 for a given tile-year"
    )
    parse.add_argument("--tile_num", help="boreal tile number", required=True)
    parse.add_argument("--year", help="atl08 year", required=True)
    parse.add_argument("--hls_path", help="HLS image path", required=True)
    parse.add_argument(
        "--topo_path", help="topo image path with slope as second band", required=True
    )
    parse.add_argument("--atl08_path", help="atl08 parquet file", required=True)
    parse.add_argument("--patch_size", help="training image patch size", type=int, default=128)
    parse.add_argument("--overlap", help="overlap between training patches", type=int, default=32)
    parse.add_argument("--rh", help="RH metric to use as learning target (Y in f^(X) = Y)", default='h_canopy')
    args = parse.parse_args()

    create_training_dataset(**vars(args))
