import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon, box

import stac_search
from build_stac_catalog import (
    _bbox_width_deg,
    build_catalog,
    build_collection,
    build_items,
    load_tile_grid,
)

NORMAL_TILE_GEOM = box(10, 50, 15, 55)
# Constructed the way a naively-reprojected antimeridian-crossing tile would look: a
# ring spanning 179 -> -179 "the wrong way" unless antimeridian.fix_shape corrects it.
DATELINE_TILE_GEOM = Polygon([(179, 60), (-179, 60), (-179, 65), (179, 65), (179, 60)])


def _write_tile_grid(path, tile_nums_and_geoms):
    gdf = gpd.GeoDataFrame(
        {'tile_num': [t for t, _ in tile_nums_and_geoms]},
        geometry=[g for _, g in tile_nums_and_geoms],
        crs='EPSG:4326',
    )
    gdf.to_file(path, driver='GPKG')
    return path


class TestBboxWidthDeg:
    def test_normal_bbox(self):
        assert _bbox_width_deg([10, 0, 15, 10]) == pytest.approx(5.0)

    def test_antimeridian_spanning_bbox(self):
        # west (179) > east (-179): a 2-degree-wide sliver straddling the dateline,
        # not the naive (and wrong) 358-degree reading.
        assert _bbox_width_deg([179, 0, -179, 10]) == pytest.approx(2.0)


class TestLoadTileGrid:
    def test_normal_tile_geometry_unaffected(self, tmp_path):
        path = _write_tile_grid(tmp_path / 'grid.gpkg', [(1, NORMAL_TILE_GEOM)])
        grid = load_tile_grid(str(path))
        assert list(grid.index) == [1]
        assert grid.loc[1, 'fixed_bbox'] == pytest.approx([10.0, 50.0, 15.0, 55.0])

    def test_dateline_crossing_tile_gets_split_and_correct_bbox_convention(
        self, tmp_path
    ):
        path = _write_tile_grid(tmp_path / 'grid.gpkg', [(2, DATELINE_TILE_GEOM)])
        grid = load_tile_grid(str(path))
        bbox = grid.loc[2, 'fixed_bbox']
        # per the GeoJSON/STAC convention, an antimeridian-spanning bbox has west > east
        assert bbox[0] > bbox[2]
        assert grid.loc[2, 'fixed_geometry']['type'] == 'MultiPolygon'


class TestBuildItems:
    def test_hls_items_have_year_in_id_and_properties(self, tmp_path):
        grid_path = _write_tile_grid(tmp_path / 'grid.gpkg', [(1, NORMAL_TILE_GEOM)])
        grid = load_tile_grid(str(grid_path))
        tindex = pd.DataFrame(
            [{'tile_num': 1, 'year': 2019, 's3_path': 's3://a/hls_1_2019.tif'}]
        )

        items = build_items(tindex, grid, 'boreal-hls-composite', has_year=True)

        assert len(items) == 1
        item = items[0]
        assert item.id == '1_2019'
        # properties also carries start_datetime/end_datetime, injected by pystac
        # itself from the constructor kwargs -- not asserting exact dict equality.
        assert item.properties['tile_num'] == 1
        assert item.properties['year'] == 2019
        assert item.assets['data'].href == 's3://a/hls_1_2019.tif'
        assert item.collection_id == 'boreal-hls-composite'
        assert item.datetime is None
        assert item.common_metadata.start_datetime.year == 2019
        assert item.common_metadata.end_datetime.year == 2019

    def test_topo_items_have_no_year_property_but_use_source_year_for_datetime(
        self, tmp_path
    ):
        grid_path = _write_tile_grid(tmp_path / 'grid.gpkg', [(1, NORMAL_TILE_GEOM)])
        grid = load_tile_grid(str(grid_path))
        tindex = pd.DataFrame([{'tile_num': 1, 's3_path': 's3://a/topo_1.tif'}])

        items = build_items(
            tindex, grid, 'boreal-topo-stack', has_year=False, source_year=2019
        )

        assert len(items) == 1
        item = items[0]
        assert item.id == '1'
        # properties also carries start_datetime/end_datetime, injected by pystac
        # itself from the constructor kwargs -- not asserting exact dict equality.
        assert item.properties['tile_num'] == 1
        assert 'year' not in item.properties
        # datetime=None + a start/end range, same convention as the per-year
        # collections -- see build_items()'s docstring.
        assert item.datetime is None
        assert item.common_metadata.start_datetime.year == 2019
        assert item.common_metadata.end_datetime.year == 2019

    def test_missing_source_year_raises_for_time_invariant_collections(self, tmp_path):
        grid_path = _write_tile_grid(tmp_path / 'grid.gpkg', [(1, NORMAL_TILE_GEOM)])
        grid = load_tile_grid(str(grid_path))
        tindex = pd.DataFrame([{'tile_num': 1, 's3_path': 's3://a/topo_1.tif'}])

        with pytest.raises(ValueError):
            build_items(tindex, grid, 'boreal-topo-stack', has_year=False)

    def test_tindex_row_referencing_unknown_tile_is_skipped_not_fatal(self, tmp_path):
        grid_path = _write_tile_grid(tmp_path / 'grid.gpkg', [(1, NORMAL_TILE_GEOM)])
        grid = load_tile_grid(str(grid_path))
        tindex = pd.DataFrame(
            [
                {'tile_num': 1, 'year': 2019, 's3_path': 's3://a/hls_1_2019.tif'},
                {'tile_num': 999, 'year': 2019, 's3_path': 's3://a/hls_999_2019.tif'},
            ]
        )

        items = build_items(tindex, grid, 'boreal-hls-composite', has_year=True)

        assert len(items) == 1
        assert items[0].id == '1_2019'


class TestBuildCollection:
    def test_raises_on_empty_items(self):
        with pytest.raises(ValueError):
            build_collection('empty-collection', [], 'description')


class TestBuildCatalog:
    def test_end_to_end_writes_catalog_and_queryable_items(self, tmp_path):
        grid_path = _write_tile_grid(
            tmp_path / 'grid.gpkg', [(1, NORMAL_TILE_GEOM), (2, DATELINE_TILE_GEOM)]
        )
        hls_path = tmp_path / 'hls_tindex.csv'
        atl08_path = tmp_path / 'atl08_tindex.csv'
        topo_path = tmp_path / 'topo_tindex.csv'
        lc_path = tmp_path / 'lc_tindex.csv'
        pd.DataFrame(
            [{'tile_num': 1, 'year': 2019, 's3_path': 's3://a/hls_1_2019.tif'}]
        ).to_csv(hls_path, index=False)
        pd.DataFrame(
            [{'tile_num': 1, 'year': 2019, 's3_path': 's3://a/atl08_1_2019.tif'}]
        ).to_csv(atl08_path, index=False)
        pd.DataFrame([{'tile_num': 1, 's3_path': 's3://a/topo_1.tif'}]).to_csv(
            topo_path, index=False
        )
        pd.DataFrame([{'tile_num': 1, 's3_path': 's3://a/lc_1.tif'}]).to_csv(
            lc_path, index=False
        )
        out_dir = tmp_path / 'catalog'

        catalog, items_path = build_catalog(
            str(hls_path),
            str(atl08_path),
            str(topo_path),
            str(lc_path),
            str(grid_path),
            str(out_dir),
        )

        assert (out_dir / 'catalog.json').exists()
        assert items_path == out_dir / 'items.parquet'
        assert items_path.exists()

        items = stac_search.load_items(str(items_path))
        assert stac_search.get_year_paths(items, 'boreal-hls-composite', 1) == {
            2019: 's3://a/hls_1_2019.tif'
        }
        assert (
            stac_search.get_single_path(items, 'boreal-topo-stack', 1)
            == 's3://a/topo_1.tif'
        )
        assert (
            stac_search.get_single_path(items, 'boreal-landcover', 1) == 's3://a/lc_1.tif'
        )
