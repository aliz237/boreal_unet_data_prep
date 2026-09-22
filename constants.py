import numpy as np


class Consts:
    # `num` is the 1-based band index in the legacy 7-band HLS composites, used only
    # as a fallback: raster_utils.resolve_band_indices prefers the band descriptions
    # carried by newer composites, whose band order differs. `alias` lists the other
    # spellings a description may use for the same band.
    HLS_BANDS = {
        'blue': {'num': 1, 'norm': None},
        'green': {'num': 2, 'norm': None},
        'red': {'num': 3, 'norm': None},
        'nir': {'num': 4, 'norm': None},
        'swir1': {'num': 5, 'norm': None, 'alias': ('swir',)},
        'swir2': {'num': 6, 'norm': None},
        # newer composites don't bound NBR: where NIR + SWIR2 approaches zero (dark
        # water and shadow, where atmospheric correction yields slightly negative
        # reflectance) the ratio blows up well past +/-1, so clip before rescaling
        # to [0, 1].
        'nbr': {'num': 7, 'norm': lambda x: (np.clip(x, -1, 1) + 1) / 2},
    }
    TOPO_BANDS = {
        'elevation': {'num': 1, 'norm': None},
        'slope': {'num': 2, 'norm': lambda x: np.clip(x, 0, 90) / Consts.MAX_SLOPE},
        'tsri': {'num': 3, 'norm': lambda x: np.clip(x, 0, 1)},
        'tpi': {'num': 4, 'norm': None},
        'slopemask': {'num': 5, 'norm': None},
    }
    # the band subsets fed to the model, shared by data_prep.py and predict.py so
    # the two can't drift apart. Order here is irrelevant -- normalize_bands emits
    # bands in HLS_BANDS/TOPO_BANDS declaration order.
    HLS_INPUT_BANDS = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'nbr']
    TOPO_INPUT_BANDS = ['slope', 'tsri']
    # normalization parameters
    MAX_SLOPE = 90.0
    MAX_HEIGHT = 100.0  # in meters
    MAX_AGB = 500.0  # in MG/ha

    # STAC collection ids, shared between build_stac_catalog.py (which builds them)
    # and data_prep.py/stac_search.py (which query them). Kept here rather than in
    # build_stac_catalog.py so data_prep.py's import graph doesn't have to pull in
    # pystac/stac-geoparquet/antimeridian just for these three strings.
    HLS_COLLECTION = 'boreal-hls-composite'
    ATL08_COLLECTION = 'boreal-atl08-labels'
    TOPO_COLLECTION = 'boreal-topo-stack'
    LC_COLLECTION = 'boreal-landcover'

    # Source product vintages for the two time-invariant collections (topo stack,
    # land cover): their tindexes carry no year column, since there's one file per
    # tile rather than per tile per year, but the underlying product still has a real
    # acquisition/reference year. build_stac_catalog.py uses these to set each
    # Item's start_datetime/end_datetime instead of an arbitrary placeholder.
    TOPO_SOURCE_YEAR = 2019  # Copernicus GLO-30
    LC_SOURCE_YEAR = 2021  # ESA WorldCover 10m v200
