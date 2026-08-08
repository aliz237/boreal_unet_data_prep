from pathlib import Path
import argparse
import logging

import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import gdal
import rasterio
from rasterio.windows import Window
import tensorflow as tf

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Consts:
    HLS_BANDS = {
        'blue': {'num': 1, 'norm': None},
        'green': {'num': 2, 'norm': None},
        'red': {'num': 3, 'norm': None},
        'nir': {'num': 4, 'norm': None},
        'swir1': {'num': 5, 'norm': None},
        'swir2': {'num': 6, 'norm': None},
        'nbr': {'num': 7, 'norm': lambda x: (x + 1) / 2},
    }
    TOPO_BANDS = {
        'elevation': {'num': 1, 'norm': None},
        'slope': {'num': 2, 'norm': lambda x: np.clip(x, 0, 90) / Consts.MAX_SLOPE},
        'tsri': {'num': 3, 'norm': lambda x: np.clip(x, 0, 1)},
        'tpi': {'num': 4, 'norm': None},
        'slopemask': {'num': 5, 'norm': None}
    }
    # normalization parameters
    MAX_SLOPE = 90.0
    MAX_HEIGHT = 100.0 # in meters
    MAX_AGB = 500.0 # in MG/ha

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
        # 'height': _int64_feature(patch_size),
        # 'width': _int64_feature(patch_size),
        # 'depth': _int64_feature(num_bands),
        'arr': _bytes_feature(arr_ser)
    }
    ex = tf.train.Example(features=tf.train.Features(feature=feature))
    return ex.SerializeToString()

def is_lidar_heavy(atl08_arr, min_valid_lidar_per_batch, nodata=-9999):
    # filtering out the low quality atl08 patches
    return np.sum(atl08_arr != nodata) > min_valid_lidar_per_batch

def gapfill(arr, max_na_block=3, nodata_thresh=0.05):
    patch_size = arr.shape[1]
    # divides patch sizes 128, 64, 32 and is large enough 
    # 2 or 4 are too small
    filter_size = 8
    bands = list(range(arr.shape[0])) # fill all bands
    # if over 5% of the Blue band is NaN, drop the patch
    ndfrac = np.isnan(arr[0]).sum() / arr[0].size
    if ndfrac > nodata_thresh:
        logger.info(f'Gapfill: Dropping low quality patch, ndfrac: {ndfrac}')
        return False

    for band in bands:
        na_blocks_band = 0
        patch_median = np.nanmedian(arr[band])
        # first try block filling
        for j in range(0, patch_size, filter_size):
            for i in range(0, patch_size, filter_size):
                win = arr[band, i:i+filter_size, j:j+filter_size]
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
            logger.info(f'Gapfill: Dropping low quality patch, na_blocks: {na_blocks_band}')
            return False

    return True

def normalize_bands(in_raster_path, out_raster_path, band_defs, band_names, mask_path=None):
    selected = {k: v for k, v in band_defs.items() if k in band_names}
    if mask_path:
        mask = rasterio.open(mask_path).read(1).astype('int32')
    with rasterio.open(in_raster_path) as src:
        arr = src.read([v['num'] for v in selected.values()]).astype('float32')
        profile = src.profile
    for i, (name, meta) in enumerate(selected.items()):
        validmask = arr[i] != -9999
        if mask_path:
            validmask &= (mask == 1)
        if meta['norm']:
            arr[i][validmask] = meta['norm'](arr[i][validmask])

    profile.update({'dtype': 'float32', 'count': len(selected), 'nodata': -9999})
    with rasterio.open(out_raster_path, 'w', **profile) as dst:
        dst.write(arr)
    return out_raster_path

def extract_patches_tfrec(
    hls_paths, # dict of {year: path} to all hls years
    atl08_paths, # dict of {year: path} to all atl08 years
    topo_path, # there's only one topo path
    tfrecord_path, # output tfrecord path
    patch_size=128,
    ndval=-9999,
    overlap=32,
    ndval_thresh=0.30
):
    years = sorted(list(hls_paths.keys()))
    step_size = patch_size - overlap
    n = 0
    min_n = int(min(patch_size * np.sqrt(2) * 0.7, 120))
    tfw = tf.io.TFRecordWriter(
        str(tfrecord_path), options=tf.io.TFRecordOptions(compression_type="GZIP")
    )
    all_dims = set()
    for t1, t2 in zip(years[:-1], years[1:]):
        logger.info(f't1:{t1}, t2:{t2}')
        with (
            rasterio.open(hls_paths[t1]) as h1,
            rasterio.open(hls_paths[t2]) as h2,
            rasterio.open(atl08_paths[t1]) as a1,
            rasterio.open(atl08_paths[t2]) as a2,
            rasterio.open(topo_path) as tp
        ):
            patch_depth = h1.count + tp.count + a1.count
            # patch_depth = 11 # 6 HLS spectral channels, 1 NBR, 1 slope, 1 TSRI, and 2 atl08 label.
            # 120 is median valid pixel count of lidar track in ATL08 128x128 patches
            # the other one is 70% of diagonal of a patch (so close to complete and decent lidar track)
            for j in range(0, h1.width - patch_size, step_size):
                for i in range(0, h1.height - patch_size, step_size):
                    # (j, i) is the top-left corner of patch
                    win = Window(j, i, patch_size, patch_size)

                    # read ATL08 patch (1-band)
                    a1_arr = a1.read(window=win).astype(np.float32)
                    a2_arr = a2.read(window=win).astype(np.float32)
                    if a1_arr.ndim != 3: # could be just RH98 lyr or include AGB lyr
                        a1_arr = a1_arr[np.newaxis, ...]
                        a2_arr = a2_arr[np.newaxis, ...]
                    # look for an ~diagonal lidar track across the patch
                    lh1 = is_lidar_heavy(a1_arr[0], min_n)
                    lh2 = is_lidar_heavy(a2_arr[0], min_n)
                    if not (is_lidar_heavy(a1_arr[0], min_n) or is_lidar_heavy(a2_arr[0], min_n)):
                        logger.debug("sparse lidar covergae, dropping patch")
                        continue
                    logger.info(f'{t1}:{lh1}, {t2}:{lh2}')
                    # read corresponding HLS patch
                    h1_arr = h1.read(window=win).astype(np.float32)
                    h2_arr = h2.read(window=win).astype(np.float32)
                    # can't have nulls in HLS, if >= ndval_thresh % is null for any band, continue
                    h1_arr[h1_arr == ndval] = np.nan
                    h2_arr[h2_arr == ndval] = np.nan
                    if np.any(
                        np.isnan(h1_arr).sum(axis=(1, 2)) >= ndval_thresh * patch_size**2
                    ) or np.any(
                        np.isnan(h2_arr).sum(axis=(1, 2)) >= ndval_thresh * patch_size**2
                    ):
                        logger.info("sparse HLS covergae, dropping patch")
                        continue
                    # fill NA
                    if not (gapfill(h1_arr) and gapfill(h2_arr)):
                        logger.info("Not gapfilling HLS, dropping patch")
                        continue
                    # same for topo
                    tp_arr = tp.read(window=win).astype(np.float32)
                    tp_arr[tp_arr == ndval] = np.nan
                    if np.any(
                        np.isnan(tp_arr).sum(axis=(1, 2)) >= ndval_thresh * patch_size**2
                    ):
                        logger.info("sparse TOPO covergae, dropping patch")
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
                    all_dims.add((h1_arr.shape, h2_arr.shape,
                                 tp_arr.shape,
                                 a1_arr.shape, a2_arr.shape,
                                 arr1.shape, arr2.shape, arr.shape))
                    # serialize the arr to write as a tfrecord
                    ser = serialize_image_patch(arr, patch_size, patch_depth)
                    tfw.write(ser.numpy())
                    if n % 100 == 0:
                        logger.info(f"wrote {n} records")
    tfw.close()
    if len(all_dims) == 1:
        logger.info(f'Shapes: {all_dims}')
    else:
        logger.info(f'shape mismatch ...')

    if n == 0:
        logger.info(f"No patches extracted from {hls_paths[t1]}!")
        tfrecord_path.unlink(missing_ok=True)
    else:
        # rename the tfrecord file to include the record count
        tfrecord_path.rename(
            tfrecord_path.with_name(
                tfrecord_path.name.replace(".tfrecord", f"_{n}.tfrecord")
            )
        )
        logger.info(f'{n} records saved')


def atl08_to_raster(atl08_path, hls_path, out_raster_path,
                    ndval=-9999, rh='h_canopy', agb=False):
    logger.info(f'out_raster_path:{out_raster_path}')
    logger.info(f'hls_path: {hls_path}')
    df = gpd.read_parquet(atl08_path)
    with rasterio.open(hls_path) as hls:
        meta = hls.meta.copy()
        h, w = hls.height, hls.width
        logger.info(f'h:{h}, w:{w}')
        rows, cols = rasterio.transform.rowcol(
            hls.transform,
            df.geometry.x.values,
            df.geometry.y.values
        )
        mask = (rows >= 0) & (cols >= 0) & (rows < h) & (cols < w)
        rows, cols = rows[mask], cols[mask]

    nlyrs = 2 if agb else 1
    out = np.full((nlyrs, h, w), ndval, dtype=np.float32)
    out[0, rows, cols] = df[rh].values[mask] / Consts.MAX_HEIGHT
    if agb:
        out[1, rows, cols] = df['AGB'].values[mask] / Consts.MAX_AGB
    meta.update({"count": nlyrs, "nodata": ndval, "dtype": "float32"})
    logger.info(out[out!=ndval][:5])
    with rasterio.open(out_raster_path, "w", **meta) as o:
        o.write(out)
        o.descriptions = (rh, 'AGB') if agb else (rh,)
        o.scales = (Consts.MAX_HEIGHT, Consts.MAX_AGB) if agb else (Consts.MAX_HEIGHT,)
        o.offsets = (0.0, 0.0) if agb else (0.0, )
        logger.info(f'wrote {out_raster_path}')


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
    dims = [ds1.RasterXSize, ds2.RasterXSize, ds3.RasterXSize,
            ds1.RasterYSize, ds2.RasterYSize, ds3.RasterYSize]
    unique_dims = len(list(set(dims)))
    if unique_dims != 1:
        ext1 = get_extent(ds1)
        ext2 = get_extent(ds2)
        ext3 = get_extent(ds3)
        logger.info('HLS, TOPO, and LC dims dont match: {dims}, resampling to common grid.')

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
        logger.info(f"Calculated Intersection: {intersection}")

    return Path(hls_path), Path(topo_path), Path(lc_path)

def reference_bounds(ref_raster):
    ref_ds = gdal.Open(ref_raster)
    gt = ref_ds.GetGeoTransform()
    xmin = gt[0]
    ymax = gt[3]
    xmax = xmin + (ref_ds.RasterXSize * gt[1])
    ymin = ymax + (ref_ds.RasterYSize * gt[5])
    bounds = (xmin, ymin, xmax, ymax)
    ref_ds = None
    return bounds

def create_fire_mask(fire_path, hls_path, year):
    bounds = reference_bounds(hls_path)
    f_df = gpd.read_file(fire_path)
    f_df = f_df[f_df.atl08_years.str.contains(str(year))]

    if f_df.shape[0] == 0:
        return None

    if fire_path.startswith('s3'):
        # save locally to DPS input dir so gdal rasterize works
        local_fire_path = Path('input')/Path(fire_path).name
        out_path = local_fire_path.with_suffix('.tif')
        f_df.to_file(str(local_fire_path))
        fire_path = local_fire_path
    else:
        out_path = Path(fire_path).with_suffix('.tif')

    ds = gdal.Rasterize(
        str(out_path),
        str(fire_path),
        format="GTiff",
        initValues=0,
        burnValues=[1],
        outputBounds=bounds,
        xRes=30,
        yRes=30,
        outputType=gdal.GDT_Int32,
        creationOptions=["COMPRESS=LZW"]
    )
    ds = None
    return out_path

def create_training_dataset(
    tile_num, atl08_tindex, hls_tindex, topo_tindex, patch_size=128,
    overlap=32, rh='h_canopy', fire_path='', agb=True, out_dir='output', ndval_thresh=0.30
):
    # if fire_path:
    #     mask_path = create_fire_mask(fire_path, hls_path, year)
    #     if mask_path is None:
    #         print(f'no fires intersect atl08-{year}, no tfrecords will be created.')
    #         return None
    # else:
    #     mask_path = None
    a = pd.read_csv(atl08_tindex)
    h = pd.read_csv(hls_tindex)
    t = pd.read_csv(topo_tindex)
    atl08_paths = a[a.tile_num==tile_num][['year', 's3_path']].set_index('year').to_dict()['s3_path']
    hls_paths = h[h.tile_num==tile_num][['year', 's3_path']].set_index('year').to_dict()['s3_path']
    topo_path = t.set_index('tile_num').loc[tile_num, 's3_path']
    logger.info(f'atl08_paths:{atl08_paths}, hls_paths:{hls_paths}, topo_path:{topo_path}')
    hls_norm_paths = dict()
    atl08_raster_paths = dict()
    logger.info("Normalizing topo")
    topo_norm_path = normalize_bands(
        topo_path,
        str(Path('/tmp')/(Path(topo_path).stem + '_norm.tif')),
        Consts.TOPO_BANDS,
        ['slope', 'tsri']
    )
    for year in atl08_paths.keys():
        atl08_raster_paths[year] = str(Path('/tmp')/(Path(atl08_paths[year]).stem + '_norm.tif'))
        logger.info(f"rasterizing atl08 {year} to HLS grid")
        atl08_to_raster(
            atl08_paths[year],
            hls_paths[year],
            atl08_raster_paths[year],
            rh=rh,
            agb=agb
        )
        logger.info(f"Normalizing HLS {year}")
        hls_norm_paths[year] = normalize_bands(
            hls_paths[year],
            str(Path('/tmp')/(Path(hls_paths[year]).stem + '_norm.tif')),
            Consts.HLS_BANDS,
            ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'nbr']
        )


    logger.info(f"Extracting patches for tile: {tile_num}")
    extract_patches_tfrec(
        hls_norm_paths,
        atl08_raster_paths,
        topo_norm_path,
        tfrecord_path=Path(f"{out_dir}/{tile_num}_2019_2024_{patch_size}_{overlap}.tfrecord.gz"),
        patch_size=patch_size,
        overlap=overlap,
        ndval_thresh=ndval_thresh
    )
    # Path(topo_norm_path).unlink(missing_ok=True)
    # for year in atl08_paths.keys():
    #     Path(hls_norm_paths.get(year)).unlink(missing_ok=True)
    #     Path(atl08_raster_paths.get(year)).unlink(missing_ok=True)


if __name__ == "__main__":
    parse = argparse.ArgumentParser(
        description="Creates a tfrecord.gz from patches of HLS, TOPO, and ATL08 for tile_num"
    )
    parse.add_argument("--tile_num", help="boreal tile number", type=int, required=True)
    parse.add_argument("--hls_tindex", help="HLS tindex path", required=True)
    parse.add_argument("--fire_path", help="fire polygon path", required=False, default='')
    parse.add_argument(
        "--topo_tindex", help="topo tindex path", required=False,
        default='s3://maap-ops-workspace/shared/montesano/DPS_tile_lists/run_build_stack_topo/build_stack_v2024_2/CopernicusGLO30/Topo_tindex_master.csv'
    )
    parse.add_argument("--atl08_tindex", help="atl08 tindex path", required=True)
    parse.add_argument("--patch_size", help="training image patch size", type=int, default=128)
    parse.add_argument("--overlap", help="overlap between training patches", type=int, default=32)
    parse.add_argument("--rh", help="RH metric to include in tfrecords", default='h_canopy')
    parse.add_argument("--out_dir", help="output directory name", default='output')
    parse.add_argument("--agb", help="Pass to include AGB layer in output tfrecords", action='store_true', default=False)
    parse.add_argument("--ndval_thresh", help="Drop the training patch if HLS nodata %% > ndval_thresh", type=float, default=0.30)

    args = parse.parse_args()

    create_training_dataset(**vars(args))
