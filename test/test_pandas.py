import pandas as pd

import njab


def test_thousands_display():
    njab.pandas.set_pandas_options()
    s = pd.Series([1_000_000])
    assert str(s)[4:13] == '1,000,000'


def test_combine_value_counts():
    df = pd.DataFrame({'a': [1, 2, 2, 2, 3, 3, 3], 'b': [1, 1, 1, 2, 2, 3, 3]})
    exp = {'a': {1: 1, 2: 3, 3: 3}, 'b': {1: 3, 2: 2, 3: 2}}
    act = njab.pandas.combine_value_counts(df).to_dict()
    assert act == exp
