import logging
import typing

import pandas as pd
import pandas.io.formats.format as pf

import omegaconf

logger = logging.getLogger(__name__)


def set_pandas_options() -> None:
    """Update default pandas options for better display."""
    pd.options.display.max_columns = 100
    pd.options.display.max_rows = 30
    pd.options.display.min_rows = 20
    pd.options.display.float_format = '{:,.3f}'.format

    class IntArrayFormatter(pf.GenericArrayFormatter):

        def _format_strings(self):
            formatter = self.formatter or '{:,d}'.format
            fmt_values = [formatter(x) for x in self.values]
            return fmt_values

    pf.IntArrayFormatter = IntArrayFormatter


def replace_with(string_key: str,
                 replace: str = "()/",
                 replace_with: str = '') -> str:
    """Replace characters in a string with a replacement."""
    for symbol in replace:
        string_key = string_key.replace(symbol, replace_with)
    return string_key


def get_colums_accessor(df: pd.DataFrame,
                        all_lower_case=False) -> omegaconf.OmegaConf:
    """Get an dictionary augmented with attribute access of column name as key
       with white spaces replaced and the original column name as values."""
    cols = {
        replace_with(col.replace(' ', '_').replace('-', '_')): col
        for col in df.columns
    }
    if all_lower_case:
        cols = {k.lower(): v for k, v in cols.items()}
    return omegaconf.OmegaConf.create(cols)


def col_isin_df(cols: typing.Union[list, str], df: pd.DataFrame) -> list:
    """Remove item (column) from passed list if not in DataFrame.
       Warning is issued for missing items.

       cols can be a comma-separated string of column names.
    """
    if isinstance(cols, str):
        cols = cols.split(',')
    ret = list()
    for _var in cols:
        if _var not in df.columns:
            logger.warning(f"Desired variable not found: {_var}", stacklevel=0)
            continue
        ret.append(_var)
    return ret


def value_counts_with_margins(y: pd.Series) -> pd.DataFrame:
    """Value counts of Series with proportion as margins."""
    ret = y.value_counts().to_frame('counts')
    ret.index.name = y.name
    ret['prop'] = y.value_counts(normalize=True)
    return ret


def get_overlapping_columns(df: pd.DataFrame, cols_expected: list) -> list:
    """Get overlapping columns between DataFrame and list of expected columns."""
    ret = df.columns.intersection(cols_expected)
    diff = pd.Index(cols_expected).difference(df.columns)
    if not diff.empty:
        logging.warning(
            f"Some columns are requested, but missing: {diff.to_list()}")
    return ret.to_list()


def combine_value_counts(X: pd.DataFrame, dropna=True) -> pd.DataFrame:
    """Pass a selection of columns to combine it's value counts.

    This performs no checks. Make sure the scale of the variables
    you pass is comparable.

    Parameters
    ----------
    X : pandas.DataFrame
        A DataFrame of several columns with values in a similar range.
    dropna : bool, optional
        Exclude NA values from counting, by default True

    Returns
    -------
    pandas.DataFrame
        DataFrame of combined value counts.
    """
    """
    """
    _df = pd.DataFrame()
    for col in X.columns:
        _df = _df.join(X[col].value_counts(dropna=dropna), how='outer')
    freq_targets = _df.sort_index()
    return freq_targets