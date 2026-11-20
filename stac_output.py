"""Write a portable DPS STAC catalog for one tile/year prediction COG."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pystac
import rasterio
import rio_stac
from pystac.extensions.raster import DataType, RasterBand, RasterExtension


def write_stac(
    cog_path,
    *,
    tile_num,
    year,
    model_path,
    agb,
    sources,
    parameters,
    model_config_path=None,
):
    """Describe a completed prediction, using only raster headers (no pixel scan).

    `sources` contains the original HLS/topo/land-cover hrefs, before downloading.
    `parameters` contains the inference settings. Model contents, sources, and
    settings identify outputs independently of temporary paths and DPS job tags.
    Call once per job, with the COG inside the DPS output directory.
    """
    cog = Path(cog_path).resolve()
    item_path = cog.with_suffix('.json')
    catalog_path = cog.parent / 'catalog.json'
    if item_path.exists() or catalog_path.exists():
        raise FileExistsError('STAC outputs already exist; use a fresh output directory')

    config_path = (
        Path(model_config_path)
        if model_config_path
        else Path(model_path).parent / 'config.json'
    )
    config_bytes = None
    config = {}
    config_output = cog.with_suffix('.model-config.json')
    if model_config_path or config_path.exists():
        config_bytes = config_path.read_bytes()
        config = json.loads(config_bytes)
        if not isinstance(config, dict):
            raise ValueError('Model config must be a JSON object')
        json.dumps(config, allow_nan=False)
        if config_output.exists():
            raise FileExistsError(f'Model config output already exists: {config_output}')

    with Path(model_path).open('rb') as model:
        model_sha256 = hashlib.file_digest(model, 'sha256').hexdigest()
    product = 'agb' if agb else 'height'
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end = datetime(year, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc)
    identity = {
        'tile_num': int(tile_num),
        'year': int(year),
        'product': product,
        'model_sha256': model_sha256,
        'sources': sources,
        'parameters': parameters,
    }
    if config_bytes is not None:
        identity['model_config_sha256'] = hashlib.sha256(config_bytes).hexdigest()
    signature = hashlib.sha256(
        json.dumps(
            identity, sort_keys=True, separators=(',', ':'), allow_nan=False
        ).encode()
    ).hexdigest()[:16]
    item_id = f'boreal-unet-{product}-{tile_num}-{year}-{signature}'
    properties = {f'boreal-unet:{key}': value for key, value in identity.items()}
    properties['boreal-unet:model_path'] = str(model_path)
    if config_bytes is not None:
        properties['boreal-unet:model_config_path'] = str(config_path)
        if config.get('run_id'):
            properties['boreal-unet:model_run_id'] = config['run_id']
    properties.update(
        start_datetime=start.isoformat().replace('+00:00', 'Z'),
        end_datetime=end.isoformat().replace('+00:00', 'Z'),
    )
    item = rio_stac.create_stac_item(
        str(cog),
        id=item_id,
        input_datetime=None,
        properties=properties,
        asset_name=product,
        asset_href=str(cog),
        asset_media_type=pystac.MediaType.COG,
        asset_roles=['data'],
        with_proj=True,
        with_raster=False,
        with_eo=False,
        geom_densify_pts=21,
    )
    # The time describes the composite year, not the prediction execution time.
    item.datetime = None
    with rasterio.open(cog) as src:
        if src.count != 1 or src.crs is None:
            raise ValueError('Expected a georeferenced, single-band prediction raster')
        band = RasterBand.create(
            data_type=DataType(src.dtypes[0]),
            nodata=src.nodata,
            unit='Mg/ha' if agb else 'm',
        )
    asset = item.assets[product]
    asset.title = 'Aboveground biomass' if agb else 'Canopy height'
    asset.description = (
        'U-Net prediction in Mg/ha' if agb else 'U-Net canopy height prediction in meters'
    )
    RasterExtension.ext(asset, add_if_missing=True).bands = [band]
    if config_bytes is not None:
        item.add_asset(
            'model_config',
            pystac.Asset(
                str(config_output),
                media_type=pystac.MediaType.JSON,
                roles=['metadata'],
                title='Model and training configuration',
            ),
        )

    # DPS supplies algorithm/version, username, job tag, created and dps-metadata.
    # Its ingestion service assigns the collection and final S3 asset locations.
    catalog = pystac.Catalog(
        id='DPS',
        description='Boreal U-Net single-tile prediction outputs',
        catalog_type=pystac.CatalogType.SELF_CONTAINED,
    )
    catalog.set_self_href(str(catalog_path))
    catalog.add_item(item)
    item.set_self_href(str(item_path))
    catalog.make_all_asset_hrefs_relative()
    catalog.validate_all()
    json.dumps(item.to_dict(), allow_nan=False)
    if config_bytes is not None:
        with config_output.open('xb') as output:
            output.write(config_bytes)
    catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
    return {
        'stac_item': str(item_path),
        'stac_catalog': str(catalog_path),
        'stac_item_id': item_id,
    }
