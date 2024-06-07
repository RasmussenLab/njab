"""Statistical functionalities for njab.

Analysis of covariance, binomial testing for binary variables,
and differential analysis of continuous variables between groups
using t-tests.
"""
from njab.stats import ancova
from njab.stats.groups_comparision import binomtest, diff_analysis

__all__ = ['ancova', 'diff_analysis', 'binomtest']
