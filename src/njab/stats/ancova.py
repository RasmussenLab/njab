"""Analysis of covariance using pingouin and statsmodels."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels


def ancova_pg(df_long: pd.DataFrame,
              feat_col: str,
              dv: str,
              between: str,
              covar: list[str] | str,
              fdr=0.05) -> pd.DataFrame:
    """ Analysis of covariance (ANCOVA) using pg.ancova
    https://pingouin-stats.org/generated/pingouin.ancova.html

    Adds multiple hypothesis testing correction by Benjamini-Hochberg
    (qvalue, rejected)

    Parameters
    ----------
    df_long : pd.DataFrame
        should be long data format
    feat_col : str
        feature column (or index) name
    dv : str
        Name of column containing the dependant variable, passed to pg.ancova
    between : str
        Name of column containing the between factor, passed to pg.ancova
    covar : list, str
        Name(s) of column(s) containing the covariate, passed to pg.ancova
    fdr : float, optional
        FDR treshold to apply, by default 0.05

    Returns
    -------
    pd.DataFrame
        Columns:  [ 'Source',
                    'SS',
                    'DF',
                    'F',
                    'p-unc',
                    'np2',
                    '{feat_col}',
                    '-Log10 pvalue',
                    'qvalue',
                    'rejected']
    """
    scores = []
    # num_covar = len(covar)

    for feat_name, data_feat in df_long.groupby(feat_col):
        # ? drop duplicated colummns in long data format?
        ancova = pg.ancova(data=data_feat, dv=dv, between=between, covar=covar)
        ancova[feat_col] = feat_name
        scores.append(ancova)
    scores = pd.concat(scores)
    scores['-Log10 pvalue'] = -np.log10(scores['p-unc'])

    return scores


def add_fdr_scores(scores: pd.DataFrame,
                   random_seed: int = None,
                   alpha: float = 0.05,
                   method: str = 'indep',
                   p_val_column: str = 'p-unc') -> pd.DataFrame:
    """Add FDR scores based on p-values in p_val_column."""
    if random_seed is not None:
        np.random.seed(random_seed)
    reject, qvalue = statsmodels.stats.multitest.fdrcorrection(
        scores[p_val_column], alpha=alpha, method=method)
    scores['qvalue'] = qvalue
    scores['rejected'] = reject
    return scores


class Ancova():
    """Base Ancova class."""

    def __init__(self,
                 df_proteomics: pd.DataFrame,
                 df_clinic: pd.DataFrame,
                 target: str,
                 covar: list[str],
                 value_name: str = 'intensity'):
        """
        Parameters
        ----------
        df_proteomics : pd.DataFrame
            proteomic measurements in wide format
        df_clinic : pd.DataFrame
            clinical data, containing `target` and `covar`
        target : str
            Variable for stratification contained in `df_clinic`
        covar : list[str]
            List of control varialbles contained in `df_clinic`
        value_name : str
            Name to be used for protemics measurements in long-format, default "intensity"
        """
        self.df_proteomics = df_proteomics
        self.df_clinic = df_clinic
        self.target = target
        self.covar = covar
        self.value_name = value_name

    def get_scores(self):
        data = (self.df_proteomics.loc[self.df_clinic[
            self.target].notna()].stack().to_frame(self.value_name).join(
                self.df_clinic))
        feat_col = data.index.names[-1]
        scores = ancova_pg(data,
                           feat_col=feat_col,
                           dv=self.value_name,
                           between=self.target,
                           covar=self.covar)
        return scores.set_index(feat_col)

    def ancova(self, random_seed=123):
        raise NotImplementedError


def filter_residuals_from_scores(scores: pd.DataFrame,
                                 filter_for='Residual') -> pd.DataFrame:
    """Remove residual from pingouin ANCOVA list."""
    scores = scores[scores.Source != filter_for]
    return scores


class AncovaAll(Ancova):
    """Ancova with FDR on all variables except the constant
       of the linear regression for each.
    """

    def ancova(self, random_seed=123):

        scores = self.get_scores()
        scores = filter_residuals_from_scores(scores)
        # drop nan values (due to multicollinearity of features - i.e. duplicated features)
        scores = scores.dropna()
        scores = add_fdr_scores(scores, random_seed=random_seed)
        self.scores = scores
        return scores.set_index('Source', append=True)


def filter_all_covars_from_scores(scores: pd.DataFrame,
                                  filter_for: str) -> pd.DataFrame:
    """Only keep feature score from pingouin ANCOVA list."""
    scores = scores[scores.Source == filter_for]
    return scores


class AncovaOnlyTarget(Ancova):
    """Ancova with FDR on only the target variables p-values
    in the set of hypothesis."""

    def ancova(self, random_seed=123) -> pd.DataFrame:
        scores = self.get_scores()
        scores = filter_all_covars_from_scores(scores, filter_for=self.target)
        scores = add_fdr_scores(scores, random_seed=random_seed)
        self.scores = scores
        return scores.set_index('Source', append=True)
