from pathlib import Path

import numpy as np
import tensorflow as tf

from patch_extraction import extract_patches_tfrec

from .helpers import write_gtiff

PATCH_SIZE = 16  # small enough for a fast test; still a multiple of gapfill's 8px filter


def _write_year(tmp_path, year, *, hls_valid=True, atl08_valid_frac=1.0, topo_valid=True):
    hls_arr = np.full((2, PATCH_SIZE, PATCH_SIZE), 0.2, dtype='float32')
    if not hls_valid:
        hls_arr[:] = -9999.0

    n_valid = int(atl08_valid_frac * PATCH_SIZE * PATCH_SIZE)
    atl08_flat = np.full(PATCH_SIZE * PATCH_SIZE, -9999.0, dtype='float32')
    atl08_flat[:n_valid] = 1.5
    atl08_arr = atl08_flat.reshape(1, PATCH_SIZE, PATCH_SIZE)

    topo_arr = np.full((1, PATCH_SIZE, PATCH_SIZE), 0.5, dtype='float32')
    if not topo_valid:
        topo_arr[:] = -9999.0

    hls_path = write_gtiff(tmp_path / f'hls_{year}.tif', hls_arr)
    atl08_path = write_gtiff(tmp_path / f'atl08_{year}.tif', atl08_arr)
    return hls_path, atl08_path, topo_arr


def test_writes_one_record_when_patch_has_sufficient_coverage(tmp_path):
    hls_2019, atl08_2019, topo_arr = _write_year(tmp_path, 2019)
    hls_2020, atl08_2020, _ = _write_year(tmp_path, 2020)
    topo_path = write_gtiff(tmp_path / 'topo.tif', topo_arr)

    out_path = Path(tmp_path / 'test_16_0.tfrecord.gz')
    extract_patches_tfrec(
        hls_paths={2019: str(hls_2019), 2020: str(hls_2020)},
        atl08_paths={2019: str(atl08_2019), 2020: str(atl08_2020)},
        topo_path=str(topo_path),
        tfrecord_path=out_path,
        patch_size=PATCH_SIZE,
        overlap=0,
    )

    written = list(tmp_path.glob('test_16_0_*.tfrecord.gz'))
    assert len(written) == 1
    assert written[0].name.endswith('_1.tfrecord.gz')

    desc = {'arr': tf.io.FixedLenFeature([], tf.string)}

    def _parse(ex):
        parsed = tf.io.parse_single_example(ex, desc)
        return tf.io.parse_tensor(parsed['arr'], out_type=tf.float32)

    ds = tf.data.TFRecordDataset(str(written[0]), compression_type='GZIP')
    records = list(ds.map(_parse))
    assert len(records) == 1
    # (2 timesteps, patch, patch, hls_bands + topo_bands + atl08_bands) = (2, 16, 16, 2+1+1)
    assert records[0].shape == (2, PATCH_SIZE, PATCH_SIZE, 4)


def test_drops_patch_and_removes_file_when_lidar_is_too_sparse(tmp_path):
    hls_2019, atl08_2019, topo_arr = _write_year(tmp_path, 2019, atl08_valid_frac=0.0)
    hls_2020, atl08_2020, _ = _write_year(tmp_path, 2020, atl08_valid_frac=0.0)
    topo_path = write_gtiff(tmp_path / 'topo.tif', topo_arr)

    out_path = Path(tmp_path / 'sparse_16_0.tfrecord.gz')
    extract_patches_tfrec(
        hls_paths={2019: str(hls_2019), 2020: str(hls_2020)},
        atl08_paths={2019: str(atl08_2019), 2020: str(atl08_2020)},
        topo_path=str(topo_path),
        tfrecord_path=out_path,
        patch_size=PATCH_SIZE,
        overlap=0,
    )

    assert not out_path.exists()
    assert list(tmp_path.glob('sparse_16_0*.tfrecord.gz')) == []


def test_drops_patch_when_hls_coverage_is_too_sparse(tmp_path):
    hls_2019, atl08_2019, topo_arr = _write_year(tmp_path, 2019, hls_valid=False)
    hls_2020, atl08_2020, _ = _write_year(tmp_path, 2020)
    topo_path = write_gtiff(tmp_path / 'topo.tif', topo_arr)

    out_path = Path(tmp_path / 'nohls_16_0.tfrecord.gz')
    extract_patches_tfrec(
        hls_paths={2019: str(hls_2019), 2020: str(hls_2020)},
        atl08_paths={2019: str(atl08_2019), 2020: str(atl08_2020)},
        topo_path=str(topo_path),
        tfrecord_path=out_path,
        patch_size=PATCH_SIZE,
        overlap=0,
    )

    assert not out_path.exists()
    assert list(tmp_path.glob('nohls_16_0*.tfrecord.gz')) == []


def test_fire_years_none_processes_every_pair(tmp_path):
    # default (non-fire) behavior with 3 years: both consecutive pairs produce a
    # record, for contrast with the fire_years-gated test below.
    hls_2019, atl08_2019, topo_arr = _write_year(tmp_path, 2019)
    hls_2020, atl08_2020, _ = _write_year(tmp_path, 2020)
    hls_2021, atl08_2021, _ = _write_year(tmp_path, 2021)
    topo_path = write_gtiff(tmp_path / 'topo.tif', topo_arr)

    out_path = Path(tmp_path / 'nofire_16_0.tfrecord.gz')
    extract_patches_tfrec(
        hls_paths={2019: str(hls_2019), 2020: str(hls_2020), 2021: str(hls_2021)},
        atl08_paths={2019: str(atl08_2019), 2020: str(atl08_2020), 2021: str(atl08_2021)},
        topo_path=str(topo_path),
        tfrecord_path=out_path,
        patch_size=PATCH_SIZE,
        overlap=0,
    )

    written = list(tmp_path.glob('nofire_16_0_*.tfrecord.gz'))
    assert len(written) == 1
    assert written[0].name.endswith('_2.tfrecord.gz')  # both pairs produced a record


def test_fire_years_skips_pairs_where_neither_year_has_fire(tmp_path):
    hls_2019, atl08_2019, topo_arr = _write_year(tmp_path, 2019)
    hls_2020, atl08_2020, _ = _write_year(tmp_path, 2020)
    hls_2021, atl08_2021, _ = _write_year(tmp_path, 2021)
    topo_path = write_gtiff(tmp_path / 'topo.tif', topo_arr)

    out_path = Path(tmp_path / 'fire_16_0.tfrecord.gz')
    extract_patches_tfrec(
        hls_paths={2019: str(hls_2019), 2020: str(hls_2020), 2021: str(hls_2021)},
        atl08_paths={2019: str(atl08_2019), 2020: str(atl08_2020), 2021: str(atl08_2021)},
        topo_path=str(topo_path),
        tfrecord_path=out_path,
        patch_size=PATCH_SIZE,
        overlap=0,
        # only 2019 has fire: (2019, 2020) is kept (2019 has fire), (2020, 2021) is
        # skipped outright (neither year has fire).
        fire_years={2019},
    )

    written = list(tmp_path.glob('fire_16_0_*.tfrecord.gz'))
    assert len(written) == 1
    assert written[0].name.endswith('_1.tfrecord.gz')
