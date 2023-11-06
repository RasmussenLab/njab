from __future__ import annotations
from matplotlib.axes import Axes
import lifelines.statistics
from lifelines import KaplanMeierFitter
import pandas as pd


def compare_km_curves(
    time: pd.Series,
    y: pd.Series,
    pred: pd.Series,
    ax: Axes = None,
    ylim: tuple[int] = (0, 1),
    xlim: tuple[int] = (0, 180),
    xlabel: str = None,
    ylabel: str = None,
    add_risk_counts=False,
) -> Axes:
    """Compare Kaplan-Meier curves for two groups (e.g. based on binary prediction)

    Parameters
    ----------
    time : pd.Series
        Time to event variable
    y : pd.Series
        event variable
    pred : pd.Series
        mask for two groups, e.g. predictions
    ax : Axes, optional
        matplotlib Axes object, by default None
    ylim : tuple[int], optional
        y-axis bounds, by default (0, 1)
    xlim : tuple[int], optional
        time-axis bounds, by default (0, 730)
    xlabel : str, optional
        time-axis label, by default None
    ylabel : str, optional
        y-axis label, by default None


    Returns
    -------
    Axes, KaplanMeierFitter, KaplanMeierFitter
        Axes object,
        KaplanMeierFitter for predited as 0,
        KaplanMeierFitter for predicted as 1

    """
    pred = pred.astype(bool)
    mask = ~pred

    kmf_0 = KaplanMeierFitter()
    kmf_0.fit(time.loc[mask], event_observed=y.loc[mask])
    ax = kmf_0.plot(xlim=xlim, ylim=ylim, legend=False, ax=ax)

    mask = pred
    kmf_1 = KaplanMeierFitter()
    kmf_1.fit(time.loc[mask], event_observed=y.loc[mask])
    ax = kmf_1.plot(xlim=xlim,
                    ylim=ylim,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    legend=False)
    if add_risk_counts:
        from lifelines.plotting import add_at_risk_counts
        add_at_risk_counts(kmf_0, kmf_1, ax=ax)
    return ax, kmf_0, kmf_1


# ! move to stats:


def log_rank_test(
    time: pd.Series,
    y: pd.Series,
    mask: pd.Series,
) -> lifelines.statistics.StatisticalResult:
    """Compare Kaplan-Meier curves for two groups (e.g. based on binary prediction).
    Given the grouping in the mask, are both KM curves significantly different?

    Parameters
    ----------
    time : pd.Series
        Time to event variable
    y : pd.Series
        event variable
    mask : pd.Series
        mask for two groups, e.g. predictions


    Returns
    -------
    StatisticalResult:
        object containing results with a wrapper for visualization
    """
    mask = mask.astype(bool)
    res = lifelines.statistics.logrank_test(
        time.loc[mask],
        time.loc[~mask],
        y.loc[mask],
        y.loc[~mask],
    )
    return res
