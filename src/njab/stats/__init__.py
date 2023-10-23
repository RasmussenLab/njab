"""Statistical functionalities for njab.

Analysis of covariance, binomial testing for binary variables,
and differential analysis of continuous variables between groups
using t-tests.
"""
from . import ancova
from .groups_comparision import diff_analysis, binomtest

__all__ = ['ancova', 'diff_analysis', 'binomtest']
