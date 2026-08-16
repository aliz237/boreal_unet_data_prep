import numpy as np
import rasterio

from raster_utils import (
    gapfill,
    is_lidar_heavy,
    normalize_bands,
    open_raster_bounds,
    raster_bounds,
)

from .helpers import make_mem_dataset, write_gdal_gtiff, write_gtiff


class TestIsLidarHeavy:
    def test_below_threshold_is_not_heavy(self):
        arr = np.full((10, 10), -9999.0)
        arr[0, :5] = 1.0  # 5 valid pixels
        # is_lidar_heavy returns numpy.bool_, not a Python bool, so compare via bool()
        # rather than `is` identity.
        assert bool(is_lidar_heavy(arr, min_valid_lidar_per_batch=5)) is False

    def test_above_threshold_is_heavy(self):
        arr = np.full((10, 10), -9999.0)
        arr[0, :6] = 1.0  # 6 valid pixels
        assert bool(is_lidar_heavy(arr, min_valid_lidar_per_batch=5)) is True


class TestGapfill:
    def test_no_nans_passes_unchanged(self):
        arr = np.arange(2 * 16 * 16, dtype='float32').reshape(2, 16, 16)
        original = arr.copy()
        assert gapfill(arr) is True
        np.testing.assert_array_equal(arr, original)

    def test_small_gap_gets_filled(self):
        rng = np.random.default_rng(0)
        arr = rng.random((1, 16, 16)).astype('float32')
        arr[0, 0:2, 0:2] = np.nan  # small gap inside one 8x8 block
        assert gapfill(arr) is True
        assert not np.isnan(arr).any()

    def test_drops_patch_when_ndfrac_too_high_in_first_band(self):
        arr = np.zeros((2, 16, 16), dtype='float32')
        arr[0, :10, :] = np.nan  # >5% of band 0 is NaN
        assert gapfill(arr, nodata_thresh=0.05) is False
        # bailed out before touching anything, band 0 NaNs are untouched
        assert np.isnan(arr[0]).sum() == 160

    def test_drops_patch_when_too_many_na_blocks(self):
        arr = np.zeros((2, 16, 16), dtype='float32')
        # band 0 stays entirely valid so the ndfrac gate passes
        # band 1: all four 8x8 blocks are fully NaN -> na_blocks_band = 4 > max_na_block
        arr[1, :, :] = np.nan
        assert gapfill(arr, max_na_block=3) is False


class TestRasterBounds:
    def test_computes_expected_bounds(self):
        ds = make_mem_dataset(
            width=10, height=5, xmin=100.0, ymax=50.0, xres=2.0, yres=3.0
        )
        xmin, ymin, xmax, ymax = raster_bounds(ds)
        assert (xmin, ymin, xmax, ymax) == (100.0, 35.0, 120.0, 50.0)

    def test_open_raster_bounds_matches_raster_bounds(self, tmp_path):
        path = write_gdal_gtiff(
            tmp_path / 'ref.tif',
            width=10,
            height=5,
            xmin=100.0,
            ymax=50.0,
            xres=2.0,
            yres=3.0,
        )
        assert open_raster_bounds(str(path)) == (100.0, 35.0, 120.0, 50.0)


class TestNormalizeBands:
    def test_applies_norm_only_to_valid_pixels_and_selects_bands(self, tmp_path):
        # band 'a' is untouched (norm=None), band 'b' is divided by 100, one pixel
        # is nodata (-9999) in both bands and must survive normalization unchanged.
        a = np.full((4, 4), 10.0, dtype='float32')
        b = np.full((4, 4), 50.0, dtype='float32')
        a[0, 0] = -9999.0
        b[0, 0] = -9999.0
        in_path = write_gtiff(tmp_path / 'in.tif', np.stack([a, b]))
        out_path = tmp_path / 'out.tif'

        band_defs = {
            'a': {'num': 1, 'norm': None},
            'b': {'num': 2, 'norm': lambda x: x / 100.0},
        }
        result = normalize_bands(str(in_path), str(out_path), band_defs, ['a', 'b'])
        assert result == str(out_path)

        with rasterio.open(out_path) as src:
            assert src.count == 2
            assert src.nodata == -9999
            out = src.read()

        assert out[0, 0, 0] == -9999.0  # band 'a' nodata pixel untouched
        assert out[1, 0, 0] == -9999.0  # band 'b' nodata pixel NOT divided by 100
        assert out[0, 1, 1] == 10.0  # band 'a' valid pixel untouched (norm=None)
        assert out[1, 1, 1] == 0.5  # band 'b' valid pixel normalized (50 / 100)

    def test_selects_only_requested_bands(self, tmp_path):
        arr = np.stack([np.full((3, 3), 1.0), np.full((3, 3), 2.0), np.full((3, 3), 3.0)])
        in_path = write_gtiff(tmp_path / 'in3.tif', arr)
        out_path = tmp_path / 'out3.tif'
        band_defs = {
            'a': {'num': 1, 'norm': None},
            'b': {'num': 2, 'norm': None},
            'c': {'num': 3, 'norm': None},
        }
        normalize_bands(str(in_path), str(out_path), band_defs, ['c'])
        with rasterio.open(out_path) as src:
            assert src.count == 1
            np.testing.assert_array_equal(
                src.read(1), np.full((3, 3), 3.0, dtype='float32')
            )
