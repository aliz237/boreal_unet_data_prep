import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from stac_search import get_single_path, get_year_paths, load_items


def _items(rows):
    return pd.DataFrame(rows)


class TestGetYearPaths:
    def test_returns_year_keyed_dict_for_matching_tile_and_collection(self):
        items = _items(
            [
                {
                    'collection': 'hls',
                    'tile_num': 10,
                    'year': 2019.0,
                    'assets': {'data': {'href': 's3://a/hls_2019.tif'}},
                },
                {
                    'collection': 'hls',
                    'tile_num': 10,
                    'year': 2020.0,
                    'assets': {'data': {'href': 's3://a/hls_2020.tif'}},
                },
                {  # different tile, must not leak in
                    'collection': 'hls',
                    'tile_num': 11,
                    'year': 2019.0,
                    'assets': {'data': {'href': 's3://a/other.tif'}},
                },
                {  # different collection, must not leak in
                    'collection': 'atl08',
                    'tile_num': 10,
                    'year': 2019.0,
                    'assets': {'data': {'href': 's3://a/atl08_2019.tif'}},
                },
            ]
        )
        result = get_year_paths(items, 'hls', 10)
        assert result == {2019: 's3://a/hls_2019.tif', 2020: 's3://a/hls_2020.tif'}
        assert all(
            isinstance(k, int) for k in result
        )  # not numpy.float64 from NaN-mixing

    def test_no_match_returns_empty_dict(self):
        items = _items(
            [
                {
                    'collection': 'hls',
                    'tile_num': 10,
                    'year': 2019.0,
                    'assets': {'data': {'href': 'x'}},
                }
            ]
        )
        assert get_year_paths(items, 'hls', 999) == {}


class TestGetSinglePath:
    def test_returns_href_for_matching_tile(self):
        items = _items(
            [
                {
                    'collection': 'topo',
                    'tile_num': 10,
                    'assets': {'data': {'href': 's3://a/topo_10.tif'}},
                },
                {
                    'collection': 'topo',
                    'tile_num': 11,
                    'assets': {'data': {'href': 's3://a/topo_11.tif'}},
                },
            ]
        )
        assert get_single_path(items, 'topo', 10) == 's3://a/topo_10.tif'

    def test_missing_tile_raises_key_error(self):
        items = _items(
            [{'collection': 'topo', 'tile_num': 10, 'assets': {'data': {'href': 'x'}}}]
        )
        with pytest.raises(KeyError):
            get_single_path(items, 'topo', 999)

    def test_duplicate_tile_raises_value_error(self):
        items = _items(
            [
                {'collection': 'topo', 'tile_num': 10, 'assets': {'data': {'href': 'a'}}},
                {'collection': 'topo', 'tile_num': 10, 'assets': {'data': {'href': 'b'}}},
            ]
        )
        with pytest.raises(ValueError):
            get_single_path(items, 'topo', 10)


class TestLoadItems:
    def test_round_trips_geoparquet(self, tmp_path):
        gdf = gpd.GeoDataFrame(
            {
                'collection': ['hls'],
                'tile_num': [10],
                'year': [2019.0],
                'assets': [{'data': {'href': 's3://a/hls_2019.tif'}}],
            },
            geometry=[box(0, 0, 1, 1)],
            crs='EPSG:4326',
        )
        path = tmp_path / 'items.parquet'
        gdf.to_parquet(path)

        items = load_items(str(path))

        assert get_year_paths(items, 'hls', 10) == {2019: 's3://a/hls_2019.tif'}
