import argparse
import logging
from pathlib import Path

import pandas as pd
import s3fs

from predict import predict_raster

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


def predict_raster_all_years(
    tile_num,
    hls_tindex,
    topo_path,
    lc_path,
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
    df = pd.read_csv(hls_tindex)
    s3 = s3fs.S3FileSystem(anon=False, client_kwargs={'region_name': 'us-west-2'})
    hls_path_year = df[df['tile_num'] == tile_num][['s3_path', 'year']].to_dict(
        orient='records'
    )
    print(hls_path_year)
    for i, item in enumerate(hls_path_year):
        try:
            logger.info(f'Downloading {item["s3_path"]}')
            local_path = str(Path(input_dir) / Path(item['s3_path']).name)
            s3.get_file(item['s3_path'], local_path)

            logger.info(f'Running predict for: {local_path}, {item["year"]}')
            out_raster_path = (
                f'{output_dir}/UNet_{Y}_m4_{patch_size}_{tile_num}_{item["year"]}.tif'
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

        except Exception as e:
            logger.info(e)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        description='Predicts a vegetation height raster given a HLS, Slope and unet model'
    )
    parse.add_argument('--tile_num', help='tile number', type=int, required=True)
    parse.add_argument(
        '--hls_tindex',
        help='HLS tindex with s3_path and tile_num attributes',
        required=True,
    )
    parse.add_argument(
        '--topo_path', help='topo image path with slope as second band', required=True
    )
    parse.add_argument('--lc_path', help='land cover image path', required=True)
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
