"""Shared test helpers for building small synthetic rasters."""

import rasterio
from osgeo import gdal
from rasterio.transform import from_origin


def write_gtiff(path, arr, nodata=-9999, dtype='float32'):
    """Writes a (count, height, width) array to a GeoTIFF at path. Returns path."""
    count, h, w = arr.shape
    profile = {
        'driver': 'GTiff',
        'height': h,
        'width': w,
        'count': count,
        'dtype': dtype,
        'crs': 'EPSG:4326',
        'transform': from_origin(0, h, 1, 1),
        'nodata': nodata,
    }
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(arr.astype(dtype))
    return path


def make_mem_dataset(width, height, xmin=0.0, ymax=10.0, xres=1.0, yres=1.0):
    """Creates an in-memory (no file) gdal.Dataset with the given geotransform."""
    ds = gdal.GetDriverByName('MEM').Create('', width, height, 1)
    ds.SetGeoTransform((xmin, xres, 0, ymax, 0, -yres))
    return ds


def write_gdal_gtiff(path, width, height, xmin=0.0, ymax=10.0, xres=1.0, yres=1.0):
    """Writes a single-band GeoTIFF to path with the given geotransform. Returns path."""
    ds = gdal.GetDriverByName('GTiff').Create(str(path), width, height, 1)
    ds.SetGeoTransform((xmin, xres, 0, ymax, 0, -yres))
    ds = None
    return path
