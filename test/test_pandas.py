import pandas as pd
import njab


def test_thousands_display():
    njab.pandas.set_pandas_options()
    s = pd.Series([1_000_000])
    assert str(s)[4:13] == '1,000,000'
