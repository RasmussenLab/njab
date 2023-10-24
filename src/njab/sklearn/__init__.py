"""Scikit-learn related functionality.

Finds the a good set of features using minimum redundancy maximum relevance (MRMR)
for a logistic regression model a binary target variable.
"""
from typing import Optional

import pandas as pd
import sklearn
import sklearn.model_selection

from mrmr import mrmr_classif

from .types import Splits, ResultsSplit, Results, AucRocCurve, PrecisionRecallCurve
from .pca import run_pca
from .preprocessing import StandardScaler
from . import scoring

__all__ = [
    'run_model',
    'get_results_split',
    'find_n_best_features',
    'run_pca',
    'scoring',
    'StandardScaler',
]

RANDOM_STATE = 42

default_log_reg = sklearn.linear_model.LogisticRegression(
    random_state=RANDOM_STATE
    #   , solver='liblinear'
)


def run_model(
    splits: Splits,
    model: sklearn.base.BaseEstimator = default_log_reg,
    fit_params=None,
    n_feat_to_select=9,
) -> Results:
    """Fit a model on the training split and calculate
       performance metrics on both train and test split.
    """
    selected_features = mrmr_classif(X=splits.X_train,
                                     y=splits.y_train,
                                     K=n_feat_to_select)

    if fit_params is None:
        fit_params = {}

    model = model.fit(splits.X_train[selected_features], splits.y_train,
                      **fit_params)

    pred_score_test = model.predict_proba(splits.X_test[selected_features])[:,
                                                                            1]
    results_test = get_results_split(y_true=splits.y_test,
                                     y_score=pred_score_test)

    pred_score_train = model.predict_proba(
        splits.X_train[selected_features])[:, 1]
    results_train = get_results_split(y_true=splits.y_train,
                                      y_score=pred_score_train)

    ret = Results(model=model,
                  selected_features=selected_features,
                  train=results_train,
                  test=results_test)
    return ret


def get_results_split(y_true, y_score):
    """Calculate metrics for a single set of samples."""
    ret = ResultsSplit(
        auc=sklearn.metrics.roc_auc_score(y_true=y_true, y_score=y_score))

    ret.roc = AucRocCurve(
        *sklearn.metrics.roc_curve(y_true=y_true, y_score=y_score))
    ret.prc = PrecisionRecallCurve(*sklearn.metrics.precision_recall_curve(
        y_true=y_true, probas_pred=y_score))

    ret.aps = sklearn.metrics.average_precision_score(y_true=y_true,
                                                      y_score=y_score)
    return ret


default_log_reg = sklearn.linear_model.LogisticRegression(
    random_state=RANDOM_STATE, solver='liblinear')


def find_n_best_features(
        X: pd.DataFrame,
        y: pd.Series,
        name: str,
        model: sklearn.base.BaseEstimator = default_log_reg,
        groups=None,  # ? Optional[array-like]
        n_features_max: int = 15,
        random_state: int = RANDOM_STATE,
        scoring: Optional[tuple] = ('precision', 'recall', 'f1',
                                    'balanced_accuracy', 'roc_auc',
                                    'average_precision'),
        return_train_score: bool = False,
        fit_params: Optional[dict] = None):
    """Create a summary of model performance on 10 times 5-fold cross-validation."""
    summary = []
    cv = sklearn.model_selection.RepeatedStratifiedKFold(
        n_splits=5, n_repeats=10, random_state=random_state)
    in_both = y.index.intersection(X.index)
    # could have a warning in case
    _X = X.loc[in_both]
    _y = y.loc[in_both]
    n_features_max = min(n_features_max, X.shape[-1])
    for n_features in range(1, n_features_max + 1):
        selected_features = mrmr_classif(_X, _y, K=n_features)
        _X_mrmr = _X[selected_features]
        scores = sklearn.model_selection.cross_validate(
            estimator=model,
            X=_X_mrmr,
            y=_y,
            groups=groups,
            scoring=scoring,
            cv=cv,
            return_train_score=return_train_score,
            fit_params=fit_params,
            error_score='raise')
        scores['n_features'] = n_features
        scores['test_case'] = name
        scores['n_observations'] = _X.shape[0]
        results = pd.DataFrame(scores)
        summary.append(results)
    summary_n_features = pd.concat(summary)
    return summary_n_features


def transform_DataFrame(X: pd.DataFrame, fct: callable) -> pd.DataFrame:
    """Set index and columns of a DataFrame after applying a callable
    which might only return a numpy array.

    Parameters
    ----------
    X : pd.DataFrame
        Original DataFrame to be transformed
    fct : callable
        Callable to be applied to every element in the DataFrame.

    Returns
    -------
    pd.DataFrame
        Transformed DataFrame
    """
    ret = fct(X)
    ret = pd.DataFrame(ret, index=X.index, columns=X.columns)
    return ret
