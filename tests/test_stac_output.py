"""DPS catalog portability, prediction identity, and raster metadata contracts."""

import json
import shutil
from pathlib import Path

import numpy as np
import pystac
import pytest
import rasterio
from rasterio.transform import from_origin

from stac_output import write_stac


@pytest.fixture
def prediction(tmp_path):
    cog = tmp_path / 'prediction.tif'
    with rasterio.open(
        cog,
        'w',
        driver='COG',
        width=16,
        height=16,
        count=1,
        dtype='float32',
        crs='EPSG:3338',
        transform=from_origin(500000, 1500000, 30, 30),
        nodata=-9999,
    ) as dst:
        dst.write(np.ones((1, 16, 16), dtype='float32'))
    model = tmp_path / 'model.keras'
    model.write_bytes(b'test model contents')
    return cog, dict(
        tile_num=3364,
        year=2024,
        model_path=model,
        agb=True,
        sources={
            'hls': 's3://bucket/hls.tif',
            'topo': 's3://bucket/topo.tif',
            'landcover': 's3://bucket/lc.tif',
        },
        parameters={
            'patch_size': 128,
            'step_size': 100,
            'ndval': -9999,
            'batch_size': 64,
            'max_na_block': 1000,
            'nodata_thresh': 0.99,
        },
    )


@pytest.fixture(autouse=True)
def local_validation(monkeypatch):
    # Unit tests exercise STAC core validation without fetching remote extension
    # schemas. Full extension validation is also exercised in the smoke check.
    def validate_core(catalog):
        for item in catalog.get_items():
            document = item.to_dict()
            document['stac_extensions'] = []
            pystac.validation.validate_dict(document)
        catalog.validate()

    monkeypatch.setattr(pystac.Catalog, 'validate_all', validate_core)


@pytest.mark.parametrize(
    'agb,asset_key,unit', [(True, 'agb', 'Mg/ha'), (False, 'height', 'm')]
)
def test_portable_catalog_and_metadata(prediction, tmp_path, agb, asset_key, unit):
    cog, kwargs = prediction
    kwargs['agb'] = agb
    result = write_stac(cog, **kwargs)
    item_json = json.loads(Path(result['stac_item']).read_text())
    assert item_json['properties']['datetime'] is None
    assert item_json['properties']['start_datetime'] == '2024-01-01T00:00:00Z'
    assert item_json['properties']['end_datetime'].startswith('2024-12-31T23:59:59')
    assert item_json['properties']['boreal-unet:tile_num'] == 3364
    assert item_json['properties']['boreal-unet:sources'] == kwargs['sources']
    assert item_json['properties']['boreal-unet:model_path'] == str(kwargs['model_path'])
    assert not any(k.startswith('maap-dps:') for k in item_json['properties'])
    assert 'created' not in item_json['properties']
    asset = item_json['assets'][asset_key]
    assert not Path(asset['href']).is_absolute()
    assert asset['type'] == pystac.MediaType.COG
    assert asset['raster:bands'] == [
        {'data_type': 'float32', 'nodata': -9999.0, 'unit': unit}
    ]
    assert item_json['properties']['proj:shape'] == [16, 16]
    assert -180 <= item_json['bbox'][0] <= 180
    assert 50 < item_json['bbox'][1] < 80

    moved = tmp_path / 'uploaded'
    moved.mkdir()
    for path in (cog, Path(result['stac_item']), Path(result['stac_catalog'])):
        shutil.move(path, moved / path.name)
    catalog = pystac.Catalog.from_file(str(moved / 'catalog.json'))
    item = next(catalog.get_items())
    assert Path(item.assets[asset_key].get_absolute_href()) == moved / cog.name
    assert item.id == result['stac_item_id']


def test_identity_is_portable_and_distinguishes_models_and_settings(prediction, tmp_path):
    cog, kwargs = prediction

    def generate(name, settings):
        directory = tmp_path / name
        directory.mkdir()
        target = directory / 'renamed.tif'
        shutil.copyfile(cog, target)
        return write_stac(target, **settings)['stac_item_id']

    original = generate('first', kwargs)
    assert generate('same', kwargs) == original
    assert generate('height', {**kwargs, 'agb': False}) != original
    assert generate('year', {**kwargs, 'year': 2023}) != original
    assert (
        generate(
            'settings',
            {**kwargs, 'parameters': {**kwargs['parameters'], 'step_size': 64}},
        )
        != original
    )
    kwargs['model_path'].write_bytes(b'different model, same filename')
    assert generate('newmodel', kwargs) != original


def test_refuses_to_overwrite_catalog(prediction):
    cog, kwargs = prediction
    catalog = cog.parent / 'catalog.json'
    catalog.write_text('existing catalog')
    with pytest.raises(FileExistsError):
        write_stac(cog, **kwargs)
    assert catalog.read_text() == 'existing catalog'


def test_single_tile_integration_preserves_source_hrefs(prediction, monkeypatch):
    import predict
    import stac_output

    cog, kwargs = prediction
    monkeypatch.setattr(
        predict, 'get_year_paths', lambda *args: {2024: kwargs['sources']['hls']}
    )
    monkeypatch.setattr(
        predict,
        'get_single_path',
        lambda _, collection, tile: {
            predict.Consts.TOPO_COLLECTION: kwargs['sources']['topo'],
            predict.Consts.LC_COLLECTION: kwargs['sources']['landcover'],
        }[collection],
    )
    from contextlib import contextmanager

    @contextmanager
    def local_copy(href, directory):
        yield '/tmp/' + href.rsplit('/', 1)[-1]

    monkeypatch.setattr(predict, 'downloaded_locally', local_copy)
    calls = []
    monkeypatch.setattr(
        predict, 'predict_raster', lambda **params: calls.append('predict')
    )

    def capture(path, **params):
        assert calls == ['predict']
        assert params['sources'] == kwargs['sources']
        assert params['parameters'] == kwargs['parameters']
        calls.append('stac')

    monkeypatch.setattr(stac_output, 'write_stac', capture)
    predict.predict_tile(
        3364,
        2024,
        None,
        kwargs['model_path'],
        str(cog),
        agb=True,
        write_stac_metadata=True,
    )
    assert calls == ['predict', 'stac']


@pytest.mark.parametrize('explicit', [False, True])
def test_preserves_model_config_with_portable_asset(prediction, tmp_path, explicit):
    import hashlib

    cog, kwargs = prediction
    config_path = tmp_path / ('training.json' if explicit else 'config.json')
    contents = b'{"run_id": "training-run-123", "TARGET": "agb", "SCALE": 500.0}\n'
    config_path.write_bytes(contents)
    if explicit:
        kwargs['model_config_path'] = config_path
    result = write_stac(cog, **kwargs)
    item_json = json.loads(Path(result['stac_item']).read_text())
    properties = item_json['properties']
    assert properties['boreal-unet:model_run_id'] == 'training-run-123'
    assert properties['boreal-unet:model_config_path'] == str(config_path)
    assert (
        properties['boreal-unet:model_config_sha256']
        == hashlib.sha256(contents).hexdigest()
    )
    asset = item_json['assets']['model_config']
    assert asset['roles'] == ['metadata']
    assert asset['type'] == 'application/json'
    assert not Path(asset['href']).is_absolute()
    snapshot = Path(result['stac_item']).parent / asset['href']
    assert snapshot.read_bytes() == contents
    config_path.write_text('{"run_id": "replacement"}')
    assert snapshot.read_bytes() == contents

    moved = tmp_path / 'uploaded'
    moved.mkdir()
    for path in (cog, snapshot, Path(result['stac_item']), Path(result['stac_catalog'])):
        shutil.move(path, moved / path.name)
    item = next(pystac.Catalog.from_file(str(moved / 'catalog.json')).get_items())
    assert Path(item.assets['model_config'].get_absolute_href()).read_bytes() == contents


def test_explicit_missing_config_fails(prediction, tmp_path):
    cog, kwargs = prediction
    with pytest.raises(FileNotFoundError):
        write_stac(cog, **kwargs, model_config_path=tmp_path / 'missing.json')
    assert not (tmp_path / 'catalog.json').exists()
