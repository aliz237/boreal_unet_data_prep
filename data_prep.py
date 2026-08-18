import argparse
import logging
from pathlib import Path

from atl08_utils import atl08_to_raster, create_fire_mask
from constants import Consts
from patch_extraction import extract_patches_tfrec
from raster_utils import normalize_bands
from stac_search import get_single_path, get_year_paths, load_items

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_training_dataset(
    tile_num,
    stac_catalog,
    patch_size=128,
    overlap=32,
    rh='h_canopy',
    fire_path='',
    agb=True,
    out_dir='output',
    ndval_thresh=0.30,
):
    items = load_items(stac_catalog)
    atl08_paths = get_year_paths(items, Consts.ATL08_COLLECTION, tile_num)
    hls_paths = get_year_paths(items, Consts.HLS_COLLECTION, tile_num)
    topo_path = get_single_path(items, Consts.TOPO_COLLECTION, tile_num)
    logger.info(
        'atl08_paths:%s, hls_paths:%s, topo_path:%s', atl08_paths, hls_paths, topo_path
    )
    hls_norm_paths = dict()
    atl08_raster_paths = dict()
    fire_mask_paths = dict()
    fire_years = set() if fire_path else None
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
            logger.info('rasterizing atl08 %s to HLS grid', year)
            atl08_to_raster(
                atl08_paths[year],
                hls_paths[year],
                atl08_raster_paths[year],
                tile_num=tile_num,
                rh=rh,
                agb=agb,
            )

            # Fire-augmentation: mask this year's HLS to only its own fire polygons
            # (via normalize_bands' mask_path), so only patches overlapping fire
            # have a chance of passing extract_patches_tfrec's nodata-fraction
            # check. A year with no matching fire is left fully unmasked, so it can
            # still serve as a clean before/after baseline for an adjacent fire
            # year -- extract_patches_tfrec's fire_years drops a (t1, t2) pair
            # outright only when *neither* year has fire coverage.
            mask_path = None
            if fire_path:
                mask_path = create_fire_mask(fire_path, hls_paths[year], year)
                if mask_path is not None:
                    fire_mask_paths[year] = mask_path
                    fire_years.add(year)
                else:
                    logger.info('no fires intersect atl08-%s for tile %s', year, tile_num)

            logger.info('Normalizing HLS %s', year)
            hls_norm_paths[year] = normalize_bands(
                hls_paths[year],
                str(Path('/tmp') / (Path(hls_paths[year]).stem + '_norm.tif')),
                Consts.HLS_BANDS,
                ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'nbr'],
                mask_path=mask_path,
            )

        logger.info('Extracting patches for tile: %s', tile_num)
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
            fire_years=fire_years,
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
            if year in fire_mask_paths:
                Path(fire_mask_paths[year]).unlink(missing_ok=True)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        description='Creates a tfrecord.gz from patches of HLS, TOPO, and ATL08 for tile_num'
    )
    parse.add_argument('--tile_num', help='boreal tile number', type=int, required=True)
    parse.add_argument(
        '--stac_catalog',
        help=(
            'path to the STAC items GeoParquet table (local path or s3://), built '
            'by build_stac_catalog.py'
        ),
        required=True,
    )
    parse.add_argument(
        '--fire_path', help='fire polygon path', required=False, default=''
    )
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
