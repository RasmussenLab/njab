import logging
import typing

import omegaconf
import pandas as pd
import pandas.io.formats.format as pf

logger = logging.getLogger(__name__)


def set_pandas_options(max_columns: int = 100,
                       max_row: int = 30,
                       min_row: int = 20,
                       float_format='{:,.3f}') -> None:
    """Update default pandas options for better display."""
    pd.options.display.max_columns = max_columns
    pd.options.display.max_rows = max_row
    pd.options.display.min_rows = min_row
    set_pandas_number_formatting(float_format=float_format)


def set_pandas_number_formatting(float_format='{:,.3f}') -> None:
    pd.options.display.float_format = float_format.format
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.describe_option.html#pandas.describe_option
    pd.options.styler.format.thousands = ','
    # # https://github.com/pandas-dev/pandas/blob/main/pandas/io/formats/format.py#L1475
    # Originally found: https://stackoverflow.com/a/29663750/9684872

    try:
        base_class = pf.GenericArrayFormatter
    except AttributeError:
        base_class = pf._GenericArrayFormatter

    class IntArrayFormatter(base_class):

        def _format_strings(self):
            formatter = self.formatter or '{:,d}'.format
            fmt_values = [formatter(x) for x in self.values]
            return fmt_values

    try:
        pf.IntArrayFormatter
        pf.IntArrayFormatter = IntArrayFormatter
    except AttributeError:
        pf._IntArrayFormatter = IntArrayFormatter


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
    freq_targets = list()
    for col in X.columns:
        freq_targets.append(X[col].value_counts(dropna=dropna).rename(col))
    freq_targets = pd.concat(freq_targets, axis=1, sort=True)
    return freq_targets
