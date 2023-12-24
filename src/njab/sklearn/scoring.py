import pandas as pd
import sklearn.metrics as sklm
import numpy as np

from .types import Results


class ConfusionMatrix():
    """Wrapper for `sklearn.metrics.confusion_matrix`"""

    def __init__(self, y_true, y_pred):
        self.cm_ = sklm.confusion_matrix(y_true, y_pred)

    def as_dataframe(self, names=('true', 'pred')) -> pd.DataFrame:
        """Create pandas.DataFrame and return.
        Names rows and columns."""
        if not hasattr(self, 'df'):
            true_name, pred_name = names
            self.df = pd.DataFrame(self.cm_)
            self.df.index.name = true_name
            self.df.columns = pd.MultiIndex.from_product([[pred_name],
                                                          self.df.columns])
        return self.df

    def classification_label(self) -> dict:
        """Classification labels as dict."""
        tn, fp, fn, tp = self.cm_.ravel()
        return {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp}

    def as_classification_series(self) -> pd.Series:
        """Classification labels as pandas.Series."""
        return pd.Series(self.classification_label())

    @property
    def as_array(self):
        """Return sklearn.metrics.confusion_matrix array"""
        return self.cm_

    def __str__(self):
        """sklearn.metrics.confusion_matrix __str__"""
        return str(self.cm_)

    def __repr__(self):
        """sklearn.metrics.confusion_matrix __repr__"""
        return repr(self.cm_)


def get_label_binary_classification(y_true: int, y_pred: int) -> str:
    """Get labels (TP, FN, TN, FP) for single case in binary classification."""
    if y_true == 1:
        if y_pred == 1:
            return 'TP'
        elif y_pred == 0:
            return 'FN'
        else:
            ValueError(f"Unknown `y_pred`: {y_pred} ({ type(y_pred) })")
    elif y_true == 0:
        if y_pred == 0:
            return 'TN'
        elif y_pred == 1:
            return 'FP'
        else:
            ValueError(f"Unknown `y_pred`: {y_pred} ({ type(y_pred) })")
    else:
        raise ValueError(f"Unknown `y_true`: {y_true} ({ type(y_pred) })")


def get_score(clf, X: pd.DataFrame, pos=1) -> pd.Series:
    """Extract score from binary classifier for class one (target class)."""
    scores = clf.predict_proba(X)
    if scores.shape[-1] > 2:
        raise NotImplementedError
    else:
        scores = scores[:, pos]
    scores = pd.Series(scores, index=X.index)
    return scores


def get_pred(clf, X: pd.DataFrame) -> pd.Series:
    """Predict class for binary classifier and keep indices of from data X."""
    ret = clf.predict(X)
    ret = pd.Series(ret, index=X.index)
    return ret


def get_custom_pred(clf, X: pd.DataFrame, cutoff=0.5) -> pd.Series:
    """Calculate predicted class for binary classifier using the specified cutoff.
       Keep indices of from data X.
    """
    scores = get_score(clf, X)
    ret = (scores > cutoff).astype(int)
    return ret


def get_target_count_per_bin(score: pd.Series,
                             y: pd.Series,
                             n_bins: int = 10) -> pd.DataFrame:
    """Created pivot table with y summed per equality sized bin of scores."""
    pred_bins = pd.DataFrame({
        'score':
        pd.cut(score, bins=list(x / n_bins for x in range(0, n_bins + 1))),
        'y==1':
        y
    })
    pred_bins = pred_bins.groupby(by='score').sum().astype(int)
    return pred_bins


def get_lr_multiplicative_decomposition(results: Results, X: pd.DataFrame,
                                        prob: pd.Series,
                                        y: pd.Series) -> pd.DataFrame:
    """Multiplicative decompositon of odds at the base of the
    logistic regresion model."""
    components = X[results.selected_features].multiply(results.model.coef_)
    components['intercept'] = float(results.model.intercept_)
    components = np.exp(components)
    components['odds'] = prob / (1.0 - prob)
    components['prob'] = prob
    components[y.name] = y
    components = components.sort_values('prob', ascending=False)
    return components
