import geopandas as gpd
import rasterio
from shapely.geometry import box

from atl08_utils import create_fire_mask

from .helpers import write_gdal_gtiff


def _write_fire_polygons(path, atl08_years, geoms):
    gdf = gpd.GeoDataFrame({'atl08_years': atl08_years}, geometry=geoms)
    gdf.to_file(path, driver='GPKG')
    return path


class TestCreateFireMask:
    def test_returns_none_when_no_fire_matches_year(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        hls_path = write_gdal_gtiff(
            tmp_path / 'hls.tif', width=2, height=2, xmin=0, ymax=60, xres=30, yres=30
        )
        fire_path = _write_fire_polygons(
            tmp_path / 'fires.gpkg', ['2019,2021'], [box(0, 0, 30, 30)]
        )

        assert create_fire_mask(str(fire_path), str(hls_path), 2020) is None

    def test_rasterizes_only_the_matching_years_fires(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # create_fire_mask hardcodes xRes=yRes=30 (real HLS pixel size), so the
        # reference raster's bounds/resolution need to actually be a multiple of
        # that for the output mask to be checkable pixel-by-pixel.
        hls_path = write_gdal_gtiff(
            tmp_path / 'hls.tif', width=2, height=2, xmin=0, ymax=60, xres=30, yres=30
        )
        # one fire matches 2019 (bottom-left 30x30 cell), one matches only 2021
        # (top-right 30x30 cell) -- only the first should be burned for year=2019.
        fire_path = _write_fire_polygons(
            tmp_path / 'fires.gpkg',
            ['2019', '2021'],
            [box(0, 0, 30, 30), box(30, 30, 60, 60)],
        )

        out_path = create_fire_mask(str(fire_path), str(hls_path), 2019)

        assert out_path is not None
        with rasterio.open(out_path) as src:
            mask = src.read(1)
        assert mask.shape == (2, 2)
        assert mask[1, 0] == 1  # bottom-left cell: the 2019 fire, burned
        assert mask[0, 1] == 0  # top-right cell: the 2021-only fire, not burned

    def test_cleans_up_its_own_intermediate_vector_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        hls_path = write_gdal_gtiff(
            tmp_path / 'hls.tif', width=2, height=2, xmin=0, ymax=60, xres=30, yres=30
        )
        fire_path = _write_fire_polygons(
            tmp_path / 'fires.gpkg', ['2019'], [box(0, 0, 30, 30)]
        )

        create_fire_mask(str(fire_path), str(hls_path), 2019)

        leftover_gpkgs = list((tmp_path / 'input').glob('*.gpkg'))
        assert leftover_gpkgs == []
