import logging
import subprocess
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from osgeo import gdal

from constants import Consts
from raster_utils import open_raster_bounds

logger = logging.getLogger(__name__)


def atl08_to_raster(
    atl08_path,
    hls_path,
    out_raster_path,
    tile_num,
    ndval=-9999,
    rh='h_canopy',
    agb=False,
):

    logger.info('out_raster_path:%s', out_raster_path)
    logger.info('hls_path: %s', hls_path)

    df = None
    atl08_agb_path = None

    try:
        if agb:
            atl08_path = Path(atl08_path)
            atl08_agb_path = Path('/tmp') / (atl08_path.stem + '_agb' + atl08_path.suffix)
            bio_models_path = Path(
                '~/Download/bio_models/bio_models_noground.tar'
            ).expanduser()

            cmd = [
                'conda',
                'run',
                '--live-stream',
                '--name',
                'data_prep2',
                'Rscript',
                'atl08_to_agb.R',
                '-a',
                str(atl08_path).replace('s3:/', '/vsis3/'),
                '-b',
                str(bio_models_path),
                '-o',
                str(atl08_agb_path),
            ]

            subprocess.run(cmd, stderr=subprocess.STDOUT, check=True)

            df = gpd.read_parquet(atl08_agb_path)
        else:
            df = gpd.read_parquet(atl08_path)

    except Exception as e:
        logger.warning(
            'Failed to read parquet file (%s). Will output an empty NoData raster.', e
        )

    has_data = df is not None and not df.empty

    # Always open HLS to get the dimensions and metadata for the output raster
    with rasterio.open(hls_path) as hls:
        meta = hls.meta.copy()
        h, w = hls.height, hls.width
        logger.info('HLS raster height:%s, width:%s', h, w)

        # Only attempt to calculate coordinates if we have a valid, populated dataframe
        if has_data:
            rows, cols = rasterio.transform.rowcol(
                hls.transform, df.geometry.x.values, df.geometry.y.values
            )
            mask = (rows >= 0) & (cols >= 0) & (rows < h) & (cols < w)
            valid_rows, valid_cols = rows[mask], cols[mask]
        else:
            logger.info('Dataframe is empty or failed to load. Skipping point mapping.')
            mask = np.array([])  # empty mask

    nlyrs = 2 if agb else 1
    # Initialize the entire array with ndval
    out = np.full((nlyrs, h, w), ndval, dtype=np.float32)

    # Map values only if we actually found valid data
    if has_data:
        out[0, valid_rows, valid_cols] = df[rh].values[mask] / Consts.MAX_HEIGHT
        if agb:
            out[1, valid_rows, valid_cols] = df['AGB'].values[mask] / Consts.MAX_AGB

    meta.update({'count': nlyrs, 'nodata': ndval, 'dtype': 'float32'})

    if df is not None and not df.empty:
        logger.info(out[out != ndval][:5])

    with rasterio.open(out_raster_path, 'w', **meta) as o:
        o.write(out)
        o.descriptions = (rh, 'AGB') if agb else (rh,)
        o.scales = (Consts.MAX_HEIGHT, Consts.MAX_AGB) if agb else (Consts.MAX_HEIGHT,)
        o.offsets = (0.0, 0.0) if agb else (0.0,)
        logger.info('wrote %s', out_raster_path)

    if agb and atl08_agb_path and atl08_agb_path.exists():
        atl08_agb_path.unlink()


def create_fire_mask(fire_path, hls_path, year):
    bounds = open_raster_bounds(hls_path)
    f_df = gpd.read_file(fire_path)
    f_df = f_df[f_df.atl08_years.str.contains(str(year))]

    if f_df.shape[0] == 0:
        return None

    if fire_path.startswith('s3'):
        # save locally to DPS input dir so gdal rasterize works
        local_fire_path = Path('input') / Path(fire_path).name
        out_path = local_fire_path.with_suffix('.tif')
        f_df.to_file(str(local_fire_path))
        fire_path = local_fire_path
    else:
        out_path = Path(fire_path).with_suffix('.tif')

    ds = gdal.Rasterize(
        str(out_path),
        str(fire_path),
        format='GTiff',
        initValues=0,
        burnValues=[1],
        outputBounds=bounds,
        xRes=30,
        yRes=30,
        outputType=gdal.GDT_Int32,
        creationOptions=['COMPRESS=LZW'],
    )
    ds = None
    return out_path
