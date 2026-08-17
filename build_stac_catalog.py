"""Builds a static STAC catalog (+ a flattened GeoParquet items table) for the HLS
composite, ATL08-derived label, and topo-stack products this pipeline consumes, from
the existing tindex CSVs and the canonical tile-grid GeoPackage.

This is a standalone, occasionally-run bootstrap script -- not part of the regular
per-tile data_prep.py run. Once built, upload out_dir's contents to S3 (e.g. via
`aws s3 sync`) and point data_prep.py's --stac_catalog at the uploaded items.parquet.

Query strategy: the catalog is queried client-side (see stac_search.py) against the
flattened GeoParquet items table, not through pystac-client search -- that requires a
live STAC API with search conformance, which a static catalog doesn't have.
"""

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import antimeridian
import geopandas as gpd
import pandas as pd
import pystac
import stac_geoparquet.arrow as sga

from constants import Consts

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

HLS_COLLECTION = Consts.HLS_COLLECTION
ATL08_COLLECTION = Consts.ATL08_COLLECTION
TOPO_COLLECTION = Consts.TOPO_COLLECTION

# The topo stack has no per-item acquisition date in its tindex. Placeholder until
# the DEM/derived-product vintage is confirmed and threaded through.
TOPO_ITEM_DATETIME = datetime(2024, 1, 1, tzinfo=timezone.utc)

# Sanity threshold for antimeridian-fix regressions: this is a circumpolar grid
# centered on the dateline, and any single ~90km tile whose fixed bbox is still this
# wide almost certainly means the fix failed rather than that the tile is real. Width
# accounts for the GeoJSON antimeridian-spanning bbox convention (west > east).
MAX_TILE_BBOX_WIDTH_DEG = 10.0

GEOTIFF_MEDIA_TYPE = 'image/tiff; application=geotiff'


def _bbox_width_deg(bbox):
    west, _, east, _ = bbox
    if west > east:  # antimeridian-spanning, per the GeoJSON/STAC bbox convention
        return (180 - west) + (east + 180)
    return east - west


def load_tile_grid(tile_grid_path):
    """Loads the canonical tile grid, reprojected to EPSG:4326 with each tile's
    geometry (and bbox) run through the antimeridian fix (this is a circumpolar grid
    centered on the dateline, so naive reprojection can produce invalid/wrapped
    geometry). Bbox is precomputed once per tile here rather than per tindex row,
    since build_items() looks up the same tile's geometry for every year it appears.
    """
    grid = gpd.read_file(tile_grid_path)[['tile_num', 'geometry']].set_index('tile_num')
    grid = grid.to_crs(4326)

    fixed_geoms = []
    fixed_bboxes = []
    for tile_num, geom in grid.geometry.items():
        fixed = antimeridian.fix_shape(geom, fix_winding=True)
        bbox = antimeridian.bbox(fixed)
        width = _bbox_width_deg(bbox)
        if width > MAX_TILE_BBOX_WIDTH_DEG:
            raise ValueError(
                f'tile_num {tile_num}: bbox width {width:.1f} deg after antimeridian '
                f'fix exceeds sanity threshold ({MAX_TILE_BBOX_WIDTH_DEG} deg) -- '
                'likely a reprojection/fix failure, not a real tile shape.'
            )
        fixed_geoms.append(fixed)
        fixed_bboxes.append(bbox)

    return grid.assign(fixed_geometry=fixed_geoms, fixed_bbox=fixed_bboxes)


def build_items(tindex_df, tile_grid, collection, has_year):
    """Converts one tindex CSV's rows into pystac.Items, joined to tile geometry by
    tile_num. `has_year=True` for HLS/ATL08 (one Item per tile per year); False for
    the time-invariant topo stack (one Item per tile).
    """
    items = []
    missing_tiles = set()
    for row in tindex_df.itertuples():
        if row.tile_num not in tile_grid.index:
            missing_tiles.add(row.tile_num)
            continue
        fixed_geometry = tile_grid.loc[row.tile_num, 'fixed_geometry']
        bbox = tile_grid.loc[row.tile_num, 'fixed_bbox']

        if has_year:
            item_id = f'{row.tile_num}_{row.year}'
            item_datetime = None
            start_datetime = datetime(row.year, 1, 1, tzinfo=timezone.utc)
            end_datetime = datetime(row.year, 12, 31, tzinfo=timezone.utc)
            properties = {'tile_num': int(row.tile_num), 'year': int(row.year)}
        else:
            item_id = f'{row.tile_num}'
            item_datetime = TOPO_ITEM_DATETIME
            start_datetime = None
            end_datetime = None
            properties = {'tile_num': int(row.tile_num)}

        items.append(
            pystac.Item(
                id=item_id,
                geometry=fixed_geometry,
                bbox=bbox,
                datetime=item_datetime,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                properties=properties,
                collection=collection,
                assets={
                    'data': pystac.Asset(href=row.s3_path, media_type=GEOTIFF_MEDIA_TYPE)
                },
            )
        )

    if missing_tiles:
        logger.warning(
            '%s: %s tindex rows reference tile_num(s) not in the tile grid, skipped: %s',
            collection,
            len(missing_tiles),
            sorted(missing_tiles),
        )
    return items


def build_collection(collection_id, items, description):
    if not items:
        raise ValueError(f'No items to build collection {collection_id!r}')

    # Naive min/max, not antimeridian-aware: fine for this grid in practice (no tile
    # actually needs splitting, verified against the real tile grid), but if some
    # future tile genuinely straddles the dateline, this collection-level extent
    # would overstate its bounds. Not used by stac_search.py's tile_num-based
    # filtering, only informational Collection metadata.
    bboxes = [item.bbox for item in items]
    overall_bbox = [
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    ]
    all_times = [item.datetime for item in items if item.datetime is not None]
    all_times += [
        item.common_metadata.start_datetime
        for item in items
        if item.common_metadata.start_datetime is not None
    ]
    all_times += [
        item.common_metadata.end_datetime
        for item in items
        if item.common_metadata.end_datetime is not None
    ]
    interval = [min(all_times), max(all_times)] if all_times else [None, None]

    collection = pystac.Collection(
        id=collection_id,
        description=description,
        extent=pystac.Extent(
            spatial=pystac.SpatialExtent(bboxes=[overall_bbox]),
            temporal=pystac.TemporalExtent(intervals=[interval]),
        ),
    )
    collection.add_items(items)
    return collection


def build_catalog(hls_tindex, atl08_tindex, topo_tindex, tile_grid_path, out_dir):
    tile_grid = load_tile_grid(tile_grid_path)

    hls_items = build_items(
        pd.read_csv(hls_tindex), tile_grid, HLS_COLLECTION, has_year=True
    )
    atl08_items = build_items(
        pd.read_csv(atl08_tindex), tile_grid, ATL08_COLLECTION, has_year=True
    )
    topo_items = build_items(
        pd.read_csv(topo_tindex), tile_grid, TOPO_COLLECTION, has_year=False
    )

    catalog = pystac.Catalog(
        id='boreal-unet-assets',
        description=(
            'HLS composite, ATL08-derived label, and topo-stack assets for the '
            'boreal UNet pipeline.'
        ),
    )
    catalog.add_child(
        build_collection(
            HLS_COLLECTION,
            hls_items,
            'Boreal HLS 3-month median composites, per tile per year.',
        )
    )
    catalog.add_child(
        build_collection(
            ATL08_COLLECTION,
            atl08_items,
            'ATL08-derived canopy height / AGB label rasters, per tile per year.',
        )
    )
    catalog.add_child(
        build_collection(
            TOPO_COLLECTION,
            topo_items,
            (
                'Copernicus GLO-30 derived topo stack (elevation, slope, TSRI, TPI, '
                'slopemask), per tile.'
            ),
        )
    )

    out_dir = Path(out_dir)
    catalog.normalize_and_save(
        str(out_dir), catalog_type=pystac.CatalogType.SELF_CONTAINED
    )
    logger.info('Wrote static catalog to %s', out_dir)

    all_items = hls_items + atl08_items + topo_items
    record_batches = sga.parse_stac_items_to_arrow(all_items)
    items_path = out_dir / 'items.parquet'
    sga.to_parquet(record_batches, str(items_path))
    logger.info('Wrote %s items to %s', len(all_items), items_path)

    return catalog, items_path


if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        description=(
            'Builds a static STAC catalog + GeoParquet items table from the existing '
            'tindex CSVs and the canonical tile grid.'
        )
    )
    parse.add_argument('--hls_tindex', help='HLS tindex CSV path', required=True)
    parse.add_argument('--atl08_tindex', help='ATL08 tindex CSV path', required=True)
    parse.add_argument('--topo_tindex', help='topo tindex CSV path', required=True)
    parse.add_argument(
        '--tile_grid', help='path to the canonical tile-grid GeoPackage', required=True
    )
    parse.add_argument(
        '--out_dir',
        help='local staging directory; upload contents to S3 separately (e.g. aws s3 sync)',
        default='stac_catalog',
    )

    args = parse.parse_args()
    build_catalog(
        args.hls_tindex, args.atl08_tindex, args.topo_tindex, args.tile_grid, args.out_dir
    )
