import argparse
from pathlib import Path
from pprint import pprint

import geopandas as gpd
import s3fs
from osgeo import gdal

from raster_utils import raster_bounds


def coin(fire_df, atl08_df, atl08_year):
    """Augments fire_df with year of coincident atl08"""
    up = False
    joined = gpd.sjoin(fire_df, atl08_df[['geometry']], how='left', predicate='contains')
    fires_with_atl08 = joined[~joined.index_right.isna()].index.unique()
    for i in fires_with_atl08:
        up = True
        fire_df.loc[i, 'coin'] = 1
        existing = fire_df.loc[i, 'atl08_years']
        fire_df.loc[i, 'atl08_years'] = (
            f'{existing},{atl08_year}' if existing else str(atl08_year)
        )
    return fire_df, up


def download_(tile_num, atl08_paths, atl08_years, hls_path):
    # download atl08 paths 2019-2024 for tile_num
    print('in download_' + '-' * 80)
    pprint(atl08_paths)
    pprint(atl08_years)
    pprint(hls_path)
    s3 = s3fs.S3FileSystem(anon=False)
    local_atl08_paths = []
    local_atl08_years = []
    for year, s3_path in zip(atl08_years, atl08_paths):
        try:
            local_path = str(Path('input') / Path(s3_path).name)
            s3.get_file(s3_path, local_path)
            local_atl08_paths.append(local_path)
            local_atl08_years.append(year)
        except Exception:
            print(f'did not find atl08 for tile, year {tile_num}-{year}')

    # need only 2019 HLS as a reference raster for CRS, shape, etc
    try:
        local_hls_path = str(Path('input') / Path(hls_path).name)
        s3.get_file(hls_path, local_hls_path)
    except Exception:
        print(f'did not find HLS for tile, year {tile_num}-2019')
        local_hls_path = None

    if not local_atl08_paths or not local_hls_path:
        raise FileNotFoundError('Either hls_path or atl08_paths dont exist')

    return local_atl08_paths, local_atl08_years, local_hls_path


def filter_fires(f_df):
    # filter fires such that either
    # a) fire_year < any atl08_year, coin = 2
    # b) atl08_year a < fire_year < atl08_year b, coin = 3
    f_df = f_df[f_df['coin'] == 1]
    f_df['yint'] = f_df['atl08_years'].apply(
        lambda years: [int(y) for y in years.split(',')]
    )
    has_post_fire = f_df['yint'].apply(max) > f_df['YEAR']
    # if atl08 year == fire_year, most likely it's a pre-fire acquisition
    has_pre_fire = f_df['yint'].apply(min) <= f_df['YEAR']
    f_df.loc[has_post_fire, 'coin'] = 2
    f_df.loc[has_pre_fire & has_post_fire, 'coin'] = 3
    f_df = f_df[f_df['coin'].isin((2, 3))]

    return f_df


def crs_bounds(hls_path):
    ref_ds = gdal.Open(hls_path)
    ref_crs = ref_ds.GetProjection()
    bounds = raster_bounds(ref_ds)
    ref_ds = None
    return ref_crs, bounds


def run_coin(fire_path, tile_num, atl08_paths, atl08_years, hls_path):
    local_atl08_paths, local_atl08_years, local_hls_path = download_(
        tile_num, atl08_paths, atl08_years, hls_path
    )
    pprint(local_atl08_paths)
    pprint(local_atl08_years)
    pprint(local_hls_path)

    ref_crs, ref_bounds = crs_bounds(local_hls_path)

    print(f'reading {fire_path}')
    f_df = gpd.read_file(fire_path)
    f_df = f_df.to_crs(ref_crs)
    f_df = gpd.clip(f_df, ref_bounds)
    print(f'clipped fire_df shape: {f_df.shape}')
    if f_df.shape[0] == 0:
        print(f'No fires in tile {tile_num}')
        return None

    f_df['coin'] = 0
    f_df['atl08_years'] = ''
    ups = []

    for a, y in zip(local_atl08_paths, local_atl08_years):
        print(
            f'finding intersections of {Path(a).stem} and {Path(fire_path).stem} for year {y}'
        )
        a_df = gpd.read_parquet(a)
        f_df, up = coin(f_df, a_df, y)
        ups.append(up)

    print(ups)
    f_df = filter_fires(f_df)

    if f_df.shape[0] > 0:
        out_fire = f'output/{Path(fire_path).stem}_{tile_num}.gpkg'
        print(f'Saving {f_df.shape[0]} fires to {out_fire}')
        f_df.to_file(out_fire)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        description='find fire polygons that intersect ATL08 tracks post fire within this tile_num'
    )
    parse.add_argument('--tile_num', help='boreal tile number', required=True)
    parse.add_argument(
        '--fire_path',
        help='fire polygons geopackage with a YEAR attribute',
        required=True,
    )
    parse.add_argument(
        '--atl08_paths',
        nargs='+',
        help='atl08 s3 paths for each year begining 2019',
        required=True,
    )
    parse.add_argument(
        '--atl08_years',
        nargs='+',
        help='atl08 years to compare against fire_years, must match the number of atl08_paths',
        required=True,
    )
    parse.add_argument('--hls_path', help='hls s3 path for year 2019', required=True)
    args = parse.parse_args()
    run_coin(**vars(args))
