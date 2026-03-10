import argparse
from pprint import pprint
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import boto3
from botocore.config import Config

def download_tfrecs(tindex_path, out_dir, workers=10):
    tindex = pd.read_csv(tindex_path)
    pprint(tindex.head())

    TFREC_DIR = Path(out_dir)
    TFREC_DIR.mkdir(exist_ok=True)

    config = Config(max_pool_connections=workers)
    client = boto3.client('s3', config=config)
    
    def download_(s3_path):
        name = Path(s3_path).name
        client.download_file(
            'maap-ops-workspace',
            s3_path.replace('s3://maap-ops-workspace/', ''),
            str(TFREC_DIR/name)
        )
        
    with ThreadPoolExecutor(max_workers=workers) as ex:
        ex.map(download_, tindex.s3_path.to_list())


if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        description="Downloads tfrecords given an s3 tindex csv"
    )
    parse.add_argument("--tindex", help="tindex csv with tile_num and s3 paths", required=True)
    parse.add_argument("--out_dir", help="local output dir for .tfrecord.gz files to be stored", required=True)
    parse.add_argument("--workers", help="tnumber of cuncurrent workers", type=int, default=10)
    args = parse.parse_args()
    download_tfrecs(**vars(args))