import numpy as np


class Consts:
    HLS_BANDS = {
        'blue': {'num': 1, 'norm': None},
        'green': {'num': 2, 'norm': None},
        'red': {'num': 3, 'norm': None},
        'nir': {'num': 4, 'norm': None},
        'swir1': {'num': 5, 'norm': None},
        'swir2': {'num': 6, 'norm': None},
        'nbr': {'num': 7, 'norm': lambda x: (x + 1) / 2},
    }
    TOPO_BANDS = {
        'elevation': {'num': 1, 'norm': None},
        'slope': {'num': 2, 'norm': lambda x: np.clip(x, 0, 90) / Consts.MAX_SLOPE},
        'tsri': {'num': 3, 'norm': lambda x: np.clip(x, 0, 1)},
        'tpi': {'num': 4, 'norm': None},
        'slopemask': {'num': 5, 'norm': None},
    }
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
