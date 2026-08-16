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
