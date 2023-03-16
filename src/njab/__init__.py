from __future__ import annotations

from importlib.metadata import version

from . import stats, sklearn, plotting

__version__ = version('njab')

__all__ = ['stats', 'sklearn', 'plotting']
