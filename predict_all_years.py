import argparse
import logging

from constants import Consts
from predict import downloaded_locally, predict_raster
from stac_search import get_single_path, get_year_paths, load_items

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


def predict_raster_all_years(
    tile_num,
    stac_catalog,
    model_path,
    output_dir='output',
    input_dir='input',
    patch_size=128,
    step_size=100,
    ndval=-9999,
    batch_size=64,
    agb=False,
    nodata_thresh=0.05,
    max_na_block=3,
):
    Y = 'AGB' if agb else 'Ht'
    items = load_items(stac_catalog)
    hls_paths = get_year_paths(items, Consts.HLS_COLLECTION, tile_num)
    topo_path = get_single_path(items, Consts.TOPO_COLLECTION, tile_num)
    lc_path = get_single_path(items, Consts.LC_COLLECTION, tile_num)
    logger.info(
        'hls_paths: %s, topo_path: %s, lc_path: %s', hls_paths, topo_path, lc_path
    )
    # topo/lc don't vary by year, so download once and reuse across the loop below,
    # cleaning up only after every year has been processed
    with (
        downloaded_locally(topo_path, input_dir) as topo_path,
        downloaded_locally(lc_path, input_dir) as lc_path,
    ):
        for year, s3_path in sorted(hls_paths.items()):
            try:
                with downloaded_locally(s3_path, input_dir) as local_path:
                    logger.info('Running predict for: %s, %s', local_path, year)
                    out_raster_path = (
                        f'{output_dir}/UNet_{Y}_m4_{patch_size}_{tile_num}_{year}.tif'
                    )
                    predict_raster(
                        hls_path=local_path,
                        topo_path=topo_path,
                        lc_path=lc_path,
                        out_raster_path=out_raster_path,
                        model_path=model_path,
                        patch_size=patch_size,
                        step_size=step_size,
                        ndval=ndval,
                        batch_size=batch_size,
                        agb=agb,
                        nodata_thresh=nodata_thresh,
                        max_na_block=max_na_block,
                    )

            except Exception:
                logger.exception(
                    'Failed to process tile %s, year %s (%s)', tile_num, year, s3_path
                )


if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        description='Predicts a vegetation height raster given a HLS, Slope and unet model'
    )
    parse.add_argument('--tile_num', help='tile number', type=int, required=True)
    parse.add_argument(
        '--stac_catalog',
        help=(
            'path to the STAC items GeoParquet table (local path or s3://), built '
            'by build_stac_catalog.py'
        ),
        required=True,
    )
    (
        parse.add_argument(
            '--output_dir',
            help='output predicted raster path',
            required=False,
            default='output',
        ),
    )
    (
        parse.add_argument(
            '--input_dir',
            help='input dir to download data',
            required=False,
            default='input',
        ),
    )
    parse.add_argument('--model_path', help='path to UNet model', required=True)
    parse.add_argument(
        '--patch_size',
        help='patch size, should be the same as what was used when training the model',
        type=int,
        default=128,
    )
    parse.add_argument(
        '--step_size',
        help='step size for sliding the window of size patch_size over the input rasters',
        type=int,
        default=100,
    )
    parse.add_argument('--ndval', help='nodata value', type=int, default=-9999)
    parse.add_argument(
        '--nodata_thresh',
        help='predict a patch if nodata fraction is below nodata_thresh',
        type=float,
        default=0.05,
    )
    parse.add_argument(
        '--max_na_block',
        help='predict a patch if < max_na_block 8x8 windows of all NaNs',
        type=int,
        default=3,
    )

    parse.add_argument(
        '--batch_size',
        help='batch size of image patches passed to model.predict',
        type=int,
        default=64,
    )
    parse.add_argument(
        '--agb',
        help='if true predict agb, o.w predict canopy height',
        action='store_true',
    )

    args = parse.parse_args()
    logger.info(args)
    predict_raster_all_years(**vars(args))
