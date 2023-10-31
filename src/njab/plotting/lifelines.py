"""
Lifeline plots. Adapted from
https://allendowney.github.io/SurvivalAnalysisPython/02_kaplan_meier.html
"""
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def plot_lifelines(obs: pd.DataFrame,
                   ax=None,
                   start_col='DateDiagnose',
                   end_col='DateDeath',
                   status_col='dead') -> matplotlib.axes.Axes:
    """Plot a lifeline for each observation.
    """
    if ax is None:
        _, ax = plt.subplots()
    for i, (_, row) in enumerate(obs.iterrows()):
        start = row[start_col]
        end = row[end_col]
        status = row[status_col]

        if not status:
            # ongoing
            ax.hlines(i, start, end, color='C2')
        else:
            # complete
            ax.hlines(i, start, end, color='C1')
            ax.plot(end, i, marker='o', color='C1')
    return ax
