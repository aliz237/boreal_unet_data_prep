from pathlib import Path
import argparse
import logging

import pandas as pd

from atl08_utils import atl08_to_raster
from constants import Consts
from patch_extraction import extract_patches_tfrec
from raster_utils import normalize_bands

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_training_dataset(
    tile_num,
    atl08_tindex,
    hls_tindex,
    topo_tindex,
    patch_size=128,
    overlap=32,
    rh='h_canopy',
    fire_path='',
    agb=True,
    out_dir='output',
    ndval_thresh=0.30,
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
    atl08_paths = (
        a[a.tile_num == tile_num][['year', 's3_path']]
        .set_index('year')
        .to_dict()['s3_path']
    )
    hls_paths = (
        h[h.tile_num == tile_num][['year', 's3_path']]
        .set_index('year')
        .to_dict()['s3_path']
    )
    topo_path = t.set_index('tile_num').loc[tile_num, 's3_path']
    logger.info(
        f'atl08_paths:{atl08_paths}, hls_paths:{hls_paths}, topo_path:{topo_path}'
    )
    hls_norm_paths = dict()
    atl08_raster_paths = dict()
    topo_norm_path = None

    try:
        logger.info('Normalizing topo')
        topo_norm_path = normalize_bands(
            topo_path,
            str(Path('/tmp') / (Path(topo_path).stem + '_norm.tif')),
            Consts.TOPO_BANDS,
            ['slope', 'tsri'],
        )
        for year in sorted(atl08_paths.keys()):
            atl08_raster_paths[year] = str(
                Path('/tmp') / (Path(atl08_paths[year]).stem + '_norm.tif')
            )
            logger.info(f'rasterizing atl08 {year} to HLS grid')
            atl08_to_raster(
                atl08_paths[year],
                hls_paths[year],
                atl08_raster_paths[year],
                tile_num=tile_num,
                rh=rh,
                agb=agb,
            )
            logger.info(f'Normalizing HLS {year}')
            hls_norm_paths[year] = normalize_bands(
                hls_paths[year],
                str(Path('/tmp') / (Path(hls_paths[year]).stem + '_norm.tif')),
                Consts.HLS_BANDS,
                ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'nbr'],
            )

        logger.info(f'Extracting patches for tile: {tile_num}')
        extract_patches_tfrec(
            hls_norm_paths,
            atl08_raster_paths,
            topo_norm_path,
            tfrecord_path=Path(
                f'{out_dir}/{tile_num}_2019_2024_{patch_size}_{overlap}.tfrecord.gz'
            ),
            patch_size=patch_size,
            overlap=overlap,
            ndval_thresh=ndval_thresh,
        )
    finally:
        logger.info('cleaning up temp rasters ...')
        if topo_norm_path:
            Path(topo_norm_path).unlink(missing_ok=True)
        for year in sorted(atl08_paths.keys()):
            if year in hls_norm_paths:
                Path(hls_norm_paths[year]).unlink(missing_ok=True)
            if year in atl08_raster_paths:
                Path(atl08_raster_paths[year]).unlink(missing_ok=True)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        description='Creates a tfrecord.gz from patches of HLS, TOPO, and ATL08 for tile_num'
    )
    parse.add_argument('--tile_num', help='boreal tile number', type=int, required=True)
    parse.add_argument('--hls_tindex', help='HLS tindex path', required=True)
    parse.add_argument(
        '--fire_path', help='fire polygon path', required=False, default=''
    )
    parse.add_argument(
        '--topo_tindex',
        help='topo tindex path',
        required=False,
        default='s3://maap-ops-workspace/shared/montesano/DPS_tile_lists/run_build_stack_topo/build_stack_v2024_2/CopernicusGLO30/Topo_tindex_master.csv',
    )
    parse.add_argument('--atl08_tindex', help='atl08 tindex path', required=True)
    parse.add_argument(
        '--patch_size', help='training image patch size', type=int, default=128
    )
    parse.add_argument(
        '--overlap', help='overlap between training patches', type=int, default=32
    )
    parse.add_argument(
        '--rh', help='RH metric to include in tfrecords', default='h_canopy'
    )
    parse.add_argument('--out_dir', help='output directory name', default='output')
    parse.add_argument(
        '--agb',
        help='Pass to include AGB layer in output tfrecords',
        action='store_true',
        default=False,
    )
    parse.add_argument(
        '--ndval_thresh',
        help='Drop the training patch if HLS nodata %% > ndval_thresh',
        type=float,
        default=0.30,
    )

    args = parse.parse_args()

    create_training_dataset(**vars(args))
