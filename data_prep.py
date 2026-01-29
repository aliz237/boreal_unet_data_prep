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
    patch_depth = 8 # 6 HLS channels, 1 slope channel, and atl08 label.
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
            slope = topo.read([2], window=win).astype(np.float32)
            slope[slope == ndval] = np.nan
            slope = np.clip(slope, 0, 90)
            slope /= 90.0
            if np.sum(np.isnan(slope)) >= ndval_thresh * patch_size**2:
                continue
            if not gapfill(slope):
                topo_patches_dropped += 1
                continue
            # save patches on disk
            n += 1
            # prep to write as TFrecord
            # concat hls, topo features and atl08 label to build one training example
            arr = np.concatenate([hls_arr, slope, lab_arr])
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


def atl08_to_raster(atl08_path, hls_path, out_raster_path, ndval=-9999):
    df = gpd.read_parquet(atl08_path)
    with rasterio.open(hls_path) as hls:
        meta = hls.meta.copy()
        h, w = hls.height, hls.width
        cols, rows = ~hls.transform * (df.geometry.x.values, df.geometry.y.values)
        cols = np.floor(cols).astype(int)
        rows = np.floor(rows).astype(int)
        mask = (rows >= 0) & (cols >= 0) & (rows < hls.height) & (cols < hls.width)
        rows, cols = rows[mask], cols[mask]
        rh98 = df["h_canopy"].values[mask]

    out = np.full((h, w), ndval, dtype=np.float32)
    out[rows, cols] = rh98
    meta.update({"count": 1, "nodata": ndval, "dtype": "float32"})
    with rasterio.open(out_raster_path, "w", **meta) as o:
        o.write(out, 1)


def subset_HLS_bands(hls_path, clean=False):
    # just get the spectral bands, no need for derived indexes like ndvi etc
    hls_path_b1_b6 = hls_path.with_name(hls_path.stem + "_b1_b6.tif")
    cmd = [
        "gdal_translate",
        "-b", "1",
        "-b", "2",
        "-b", "3",
        "-b", "4",
        "-b", "5",
        "-b", "6",
        str(hls_path),
        str(hls_path_b1_b6),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if clean:
        hls_path.unlink()

    return hls_path_b1_b6


def get_extent(ds):
    gt = ds.GetGeoTransform()
    width = ds.RasterXSize
    height = ds.RasterYSize

    xmin = gt[0]
    ymax = gt[3]
    xmax = xmin + (width * gt[1])
    ymin = ymax + (height * gt[5])

    return [xmin, ymin, xmax, ymax]


def align_if_needed(hls_path, topo_path):
    ds1 = gdal.Open(hls_path)
    ds2 = gdal.Open(topo_path)

    if (ds1.RasterXSize != ds2.RasterXSize) or (ds1.RasterYSize != ds2.RasterYSize):
        ext1 = get_extent(ds1)
        ext2 = get_extent(ds2)

        intersection = [
            max(ext1[0], ext2[0]),
            max(ext1[1], ext2[1]),
            min(ext1[2], ext2[2]),
            min(ext1[3], ext2[3])
        ]

        warp_options = gdal.WarpOptions(
            outputBounds=intersection,
            width=3000,
            height=3000,
            resampleAlg='bilinear',
            format='GTiff',
            srcNodata=-9999,
            dstNodata=-9999
        )
        hls_path = Path(hls_path)
        hls_path = hls_path.with_name(hls_path.name.replace('.tif', '_resamp.tif'))
        topo_path = Path(topo_path)
        topo_path = topo_path.with_name(topo_path.name.replace('.tif', '_resamp.tif'))

        gdal.Warp(str(hls_path), ds1, options=warp_options)
        gdal.Warp(str(topo_path), ds2, options=warp_options)

        ds1 = ds2 = None
        print(f"Calculated Intersection: {intersection}")

    return Path(hls_path), Path(topo_path)


def create_training_dataset(
    tile_num, year, atl08_path, hls_path, slope_path, patch_size=128, overlap=32
):

    print("rasterizing atl08 to HLS grid")
    atl08_path = Path(atl08_path)
    hls_path = Path(hls_path)

    atl08_raster_path = atl08_path.with_suffix(".tif")
    atl08_to_raster(str(atl08_path), str(hls_path), str(atl08_raster_path))

    print("subsetting [B, G, R, NIR, SWIR1, SWIR2] from HLS bands")
    hls_path_b1_b6 = subset_HLS_bands(hls_path, clean=False)

    print(f"Extracting patches for tile-year: {tile_num}-{year}")
    extract_patches_tfrec(
        hls_path_b1_b6,
        atl08_raster_path,
        slope_path,
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
        "--slope_path", help="topo image path with slope as second band", required=True
    )
    parse.add_argument("--atl08_path", help="atl08 parquet file", required=True)
    parse.add_argument("--patch_size", help="training image patch size", type=int, default=128)
    parse.add_argument("--overlap", help="overlap between training patches", type=int, default=32)
    args = parse.parse_args()

    create_training_dataset(**vars(args))
