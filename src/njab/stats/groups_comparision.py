"""Bionomial test and t-test for groups comparision."""
from __future__ import annotations
import logging
import pandas as pd
import pingouin as pg

from scipy.stats import binomtest as scipy_binomtest

logger = logging.getLogger(__name__)


def means_between_groups(
    df: pd.DataFrame,
    boolean_array: pd.Series,
    event_names: tuple[str, str] = ('1', '0')
) -> pd.DataFrame:
    """Mean comparison between groups"""
    sub = df.loc[boolean_array].describe().iloc[:3]
    sub['event'] = event_names[0]
    sub = sub.set_index('event', append=True).swaplevel()
    ret = sub
    sub = df.loc[~boolean_array].describe().iloc[:3]
    sub['event'] = event_names[1]
    sub = sub.set_index('event', append=True).swaplevel()
    ret = pd.concat([ret, sub])
    ret.columns.name = 'variable'
    ret.index.names = ('event', 'stats')
    return ret.T


def calc_stats(df: pd.DataFrame, boolean_array: pd.Series,
               vars: list[str]) -> pd.DataFrame:
    """Calculate t-test for each variable in `vars` between two groups defined
    by boolean array."""
    ret = []
    for var in vars:
        _ = pg.ttest(df.loc[boolean_array, var], df.loc[~boolean_array, var])
        ret.append(_)
    ret = pd.concat(ret)
    ret = ret.set_index(vars)
    ret.columns.name = 'ttest'
    ret.columns = pd.MultiIndex.from_product([['ttest'], ret.columns],
                                             names=('test', 'var'))
    return ret


def diff_analysis(
    df: pd.DataFrame,
    boolean_array: pd.Series,
    event_names: tuple[str, str] = ('1', '0'),
    ttest_vars=("alternative", "p-val", "cohen-d")
) -> pd.DataFrame:
    """Differential analysis procedure between two groups. Calculaes
    mean per group and t-test for each variable in `vars` between two groups."""
    ret = means_between_groups(df,
                               boolean_array=boolean_array,
                               event_names=event_names)
    ttests = calc_stats(df, boolean_array=boolean_array, vars=ret.index)
    ret = ret.join(ttests.loc[:, pd.IndexSlice[:, ttest_vars]])
    return ret


def binomtest(
    var: pd.Series,
    boolean_array: pd.Series,
    alternative='two-sided',
    event_names: tuple[str, str] = ('event', 'no-event')
) -> pd.DataFrame:
    """Binomial test for categorical variable between two groups defined by a
    boolean array."""
    entry = {}
    entry['variable'] = var.name

    if var.dtype != 'category':
        logger.warn(
            f"Passed on categorical data (which was expected): {var.name}")
        var = var.astype('category')

    assert len(
        var.cat.categories
    ) == 2, f"No binary variable, found {len(var.cat.categories)} categories: {list(var.cat.categories)}"

    p_1 = var.loc[boolean_array].dropna().cat.codes.mean()

    p_0 = var.loc[~boolean_array].dropna().cat.codes.mean()
    logger.debug(f"p cat==0: {p_0}, p cat==1: {p_1}")

    cat_at_pos_one = var.cat.categories[1]
    logger.debug('Category with code 1', cat_at_pos_one)

    counts = var.loc[boolean_array].value_counts()
    k, n = counts.loc[cat_at_pos_one], counts.sum()

    entry[event_names[0]] = dict(count=n, p=p_1)
    entry[event_names[1]] = dict(
        count=var.loc[~boolean_array].value_counts().sum(), p=p_0)

    test_res = scipy_binomtest(k, n, p_0, alternative=alternative)
    test_res = pd.Series(test_res.__dict__).to_frame('binomial test').unstack()
    test_res.name = entry['variable']
    test_res = test_res.to_frame().T

    entry = pd.DataFrame(entry).set_index('variable', append=True).unstack(0)
    entry = entry.join(test_res)
    return entry
