"""not just another biomarker (njab) - a package for biomarker discovery

Provides an opinionated workflow for biomarker discovery, using
mimimum redundancy maximum relevance (MRMR) feature selection and
logistic regression as a simple and explainable model.
"""
from importlib.metadata import version

from njab import io, pandas, plotting, sklearn, stats

__version__ = version('njab')

__all__ = ['stats', 'sklearn', 'plotting', 'pandas', 'io']
