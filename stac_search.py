"""Runtime search helper against the flattened STAC items GeoParquet table built by
build_stac_catalog.py. Client-side filtering (no live STAC API) -- see that module's
docstring for why.
"""

import geopandas as gpd


def load_items(stac_catalog_path):
    """Loads the flattened STAC items table (local path or s3:// URL)."""
    return gpd.read_parquet(stac_catalog_path)


def _asset_href(assets, asset_key='data'):
    return assets[asset_key]['href']


def get_year_paths(items, collection, tile_num):
    """Returns {year: asset_href} for every Item in `collection` matching `tile_num`.

    Mirrors the {year: s3_path} shape create_training_dataset previously built from
    a tindex CSV via `.set_index('year').to_dict()['s3_path']`.
    """
    matches = items[(items['collection'] == collection) & (items['tile_num'] == tile_num)]
    # `year` comes back as float64 from the GeoParquet round-trip (other collections'
    # items have no `year` property, so the shared column is NaN-mixed); cast back to
    # int to match the {int: str} shape create_training_dataset expects.
    return {int(row.year): _asset_href(row.assets) for row in matches.itertuples()}


def get_single_path(items, collection, tile_num):
    """Returns the asset_href for the one Item in `collection` matching `tile_num`.

    For time-invariant collections (e.g. topo) where there's exactly one Item per
    tile. Mirrors today's `t.set_index('tile_num').loc[tile_num, 's3_path']`.
    """
    matches = items[(items['collection'] == collection) & (items['tile_num'] == tile_num)]
    if matches.empty:
        raise KeyError(f'No {collection} item found for tile_num {tile_num}')
    if len(matches) > 1:
        raise ValueError(
            f'Expected exactly one {collection} item for tile_num {tile_num}, '
            f'found {len(matches)}'
        )
    return _asset_href(matches.iloc[0].assets)
