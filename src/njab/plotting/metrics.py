import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from njab.sklearn.types import ResultsSplit, Results


def plot_split_auc(result: ResultsSplit, name: str,
                   ax: matplotlib.axes.Axes) -> matplotlib.axes.Axes:
    col_name = f"{name} (auc: {result.auc:.3f})"
    roc = pd.DataFrame(result.roc, index='fpr tpr cutoffs'.split()).rename(
        {'tpr': col_name})
    ax = roc.T.plot('fpr',
                    col_name,
                    xlabel='false positive rate',
                    ylabel='true positive rate',
                    style='.-',
                    ax=ax)
    return ax


def plot_auc(results: Results,
             ax: matplotlib.axes.Axes = None,
             **kwargs) -> matplotlib.axes.Axes:
    if ax is None:
        fig, ax = plt.subplots(1, 1, **kwargs)
    ax = plot_split_auc(results.train, f"{results.name} (train)", ax)
    ax = plot_split_auc(results.test, f"{results.name} (test)", ax)
    return ax


def plot_split_prc(result: ResultsSplit, name: str,
                   ax: matplotlib.axes.Axes) -> matplotlib.axes.Axes:
    col_name = f"{name} (aps: {result.aps:.3f})"
    roc = pd.DataFrame(result.prc,
                       index='precision recall cutoffs'.split()).rename(
                           {'precision': col_name})
    ax = roc.T.plot('recall',
                    col_name,
                    xlabel='true positive rate',
                    ylabel='precision',
                    style='.-',
                    ax=ax)
    return ax


def plot_prc(results: ResultsSplit, ax: matplotlib.axes.Axes = None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, **kwargs)
    ax = plot_split_prc(results.train, f"{results.name} (train)", ax)
    ax = plot_split_prc(results.test, f"{results.name} (test)", ax)
    return ax