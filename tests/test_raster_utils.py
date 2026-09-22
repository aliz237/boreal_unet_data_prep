import numpy as np
import pytest
import rasterio

from constants import Consts
from raster_utils import (
    gapfill,
    is_lidar_heavy,
    normalize_bands,
    open_raster_bounds,
    raster_bounds,
    resolve_band_indices,
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

    def test_all_nan_block_falls_back_to_patch_median(self):
        arr = np.zeros((1, 16, 16), dtype='float32')
        arr[0, 8:, 8:] = 4.0
        arr[0, :8, :8] = np.nan
        assert gapfill(arr, max_na_block=1, nodata_thresh=0.99) is True
        np.testing.assert_array_equal(arr[0, :8, :8], np.zeros((8, 8), dtype='float32'))

    def test_bands_are_filled_independently(self):
        arr = np.stack([np.full((16, 16), 1.0), np.full((16, 16), 7.0)]).astype('float32')
        arr[:, 0, 0] = np.nan
        assert gapfill(arr) is True
        assert arr[0, 0, 0] == 1.0
        assert arr[1, 0, 0] == 7.0


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

    def test_open_raster_bounds_works_even_when_gdal_open_would_fail(
        self, tmp_path, monkeypatch
    ):
        # gdal.Open() doesn't reliably handle s3:// paths -- it returns None instead
        # of raising. Simulate that failure and confirm open_raster_bounds (which
        # uses rasterio instead) doesn't depend on gdal.Open's result.
        path = write_gdal_gtiff(
            tmp_path / 'ref.tif',
            width=10,
            height=5,
            xmin=100.0,
            ymax=50.0,
            xres=2.0,
            yres=3.0,
        )
        monkeypatch.setattr('raster_utils.gdal.Open', lambda *a, **k: None)
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

    def test_mask_path_nulls_out_pixels_even_without_norm(self, tmp_path):
        # A band with norm=None (most HLS bands) must still get masked-out pixels
        # set to nodata, even though the mask never runs through norm().
        a = np.full((4, 4), 10.0, dtype='float32')
        in_path = write_gtiff(tmp_path / 'in.tif', np.stack([a]))
        out_path = tmp_path / 'out.tif'

        mask = np.zeros((4, 4), dtype='float32')
        mask[0:2, :] = 1  # top half "in fire", bottom half not
        mask_path = write_gtiff(tmp_path / 'mask.tif', np.stack([mask]))

        band_defs = {'a': {'num': 1, 'norm': None}}
        normalize_bands(
            str(in_path), str(out_path), band_defs, ['a'], mask_path=str(mask_path)
        )

        with rasterio.open(out_path) as src:
            out = src.read(1)
        assert (out[0:2, :] == 10.0).all()  # inside mask: untouched (norm=None)
        assert (out[2:4, :] == -9999.0).all()  # outside mask: nulled to nodata

    def test_mask_path_nulls_out_pixels_with_norm(self, tmp_path):
        b = np.full((4, 4), 50.0, dtype='float32')
        in_path = write_gtiff(tmp_path / 'in.tif', np.stack([b]))
        out_path = tmp_path / 'out.tif'

        mask = np.zeros((4, 4), dtype='float32')
        mask[0:2, :] = 1
        mask_path = write_gtiff(tmp_path / 'mask.tif', np.stack([mask]))

        band_defs = {'b': {'num': 1, 'norm': lambda x: x / 100.0}}
        normalize_bands(
            str(in_path), str(out_path), band_defs, ['b'], mask_path=str(mask_path)
        )

        with rasterio.open(out_path) as src:
            out = src.read(1)
        assert (out[0:2, :] == 0.5).all()  # inside mask: normalized
        # outside mask: nulled, not left as the un-normalized raw value (50.0)
        assert (out[2:4, :] == -9999.0).all()


class TestResolveBandIndices:
    def test_falls_back_to_num_without_descriptions(self):
        selected = {'a': {'num': 3, 'norm': None}, 'b': {'num': 1, 'norm': None}}
        assert resolve_band_indices((None, None, None), selected) == [3, 1]

    def test_prefers_descriptions_over_num(self):
        selected = {'a': {'num': 1, 'norm': None}, 'b': {'num': 2, 'norm': None}}
        assert resolve_band_indices(('b', 'c', 'a'), selected) == [3, 1]

    def test_matching_ignores_case_and_punctuation(self):
        selected = {'swir2': {'num': 6, 'norm': None}}
        assert resolve_band_indices(('SWIR_2',), selected) == [1]

    def test_uses_alias(self):
        selected = {'swir1': {'num': 5, 'norm': None, 'alias': ('swir',)}}
        assert resolve_band_indices(('Blue', 'SWIR'), selected) == [2]

    def test_raises_when_described_raster_lacks_band(self):
        selected = {'nbr': {'num': 7, 'norm': None}}
        with pytest.raises(KeyError, match='nbr'):
            resolve_band_indices(('blue', 'green'), selected)

    def test_resolves_new_hls_band_order(self):
        # the 21-band composites carry every required band, but NBR sits at 12
        # rather than 7 and SWIR1 is spelled 'SWIR'.
        descriptions = (
            'Blue', 'Green', 'Red', 'NIR', 'SWIR', 'SWIR2', 'NDVI', 'SAVI',
            'MSAVI', 'NDMI', 'EVI', 'NBR', 'NBR2', 'TCB', 'TCG', 'TCW',
            'ValidMask', 'Xgeo', 'Ygeo', 'JulianDate', 'yearDate',
        )
        selected = {
            k: v for k, v in Consts.HLS_BANDS.items() if k in Consts.HLS_INPUT_BANDS
        }
        assert resolve_band_indices(descriptions, selected) == [1, 2, 3, 4, 5, 6, 12]


class TestNormalizeBandsWithDescriptions:
    def test_reads_by_description_and_writes_in_band_defs_order(self, tmp_path):
        arr = np.stack(
            [np.full((3, 3), float(i), dtype='float32') for i in range(1, 4)]
        )
        in_path = write_gtiff(
            tmp_path / 'desc.tif', arr, descriptions=['c', 'a', 'b']
        )
        out_path = tmp_path / 'desc_norm.tif'
        band_defs = {
            'a': {'num': 1, 'norm': None},
            'b': {'num': 2, 'norm': None},
            'c': {'num': 3, 'norm': None},
        }
        normalize_bands(str(in_path), str(out_path), band_defs, ['a', 'b', 'c'])

        with rasterio.open(out_path) as src:
            assert src.descriptions == ('a', 'b', 'c')
            out = src.read()
        # output order follows band_defs, values follow the source descriptions
        assert [out[i, 0, 0] for i in range(3)] == [2.0, 3.0, 1.0]

    def test_nbr_norm_clips_out_of_range_values(self):
        nbr_norm = Consts.HLS_BANDS['nbr']['norm']
        arr = np.array([-129.0, -1.0, 0.0, 1.0, 14.0], dtype='float32')
        np.testing.assert_allclose(nbr_norm(arr), [0.0, 0.0, 0.5, 1.0, 1.0])
