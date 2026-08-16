import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

from coincident_fire_atl08 import coin, filter_fires


def _fire_df():
    return gpd.GeoDataFrame(
        {
            'name': ['A', 'B'],
            'coin': [0, 0],
            'atl08_years': ['', ''],
        },
        geometry=[box(0, 0, 10, 10), box(20, 20, 30, 30)],
        crs='EPSG:4326',
    )


def _atl08_df(*points):
    return gpd.GeoDataFrame(geometry=list(points), crs='EPSG:4326')


class TestCoin:
    def test_marks_fire_containing_an_atl08_point(self):
        fire_df = _fire_df()
        atl08_df = _atl08_df(Point(5, 5))  # inside fire 'A' only

        result, up = coin(fire_df, atl08_df, 2019)

        assert up is True
        assert result.loc[0, 'coin'] == 1
        assert result.loc[0, 'atl08_years'] == '2019'
        assert result.loc[1, 'coin'] == 0
        assert result.loc[1, 'atl08_years'] == ''

    def test_no_intersection_leaves_fire_df_unchanged(self):
        fire_df = _fire_df()
        atl08_df = _atl08_df(Point(1000, 1000))  # inside neither fire

        result, up = coin(fire_df, atl08_df, 2019)

        assert up is False
        assert (result['coin'] == 0).all()
        assert (result['atl08_years'] == '').all()

    def test_accumulates_years_across_calls(self):
        fire_df = _fire_df()
        fire_df, _ = coin(fire_df, _atl08_df(Point(5, 5)), 2019)
        fire_df, _ = coin(fire_df, _atl08_df(Point(6, 6)), 2021)

        assert fire_df.loc[0, 'coin'] == 1
        assert fire_df.loc[0, 'atl08_years'] == '2019,2021'


class TestFilterFires:
    def test_keeps_only_pre_and_post_fire_patterns(self):
        f_df = pd.DataFrame(
            {
                'coin': [0, 1, 1, 1, 1],
                'atl08_years': ['', '2019', '2021', '2019,2021', '2020'],
                'YEAR': [2020, 2020, 2020, 2020, 2020],
            }
        )

        result = filter_fires(f_df)

        # row 0: never coincident -> dropped upfront
        # row 1: pre-fire only ('2019' < 2020) -> not post-fire, dropped
        # row 2: post-fire only ('2021' > 2020) -> kept, coin=2
        # row 3: both pre and post -> kept, coin=3
        # row 4: atl08 year == fire year counts as pre-fire only -> dropped
        assert sorted(result.index.tolist()) == [2, 3]
        assert result.loc[2, 'coin'] == 2
        assert result.loc[3, 'coin'] == 3
