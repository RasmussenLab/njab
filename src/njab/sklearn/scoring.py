import pandas as pd
import sklearn.metrics as sklm


class ConfusionMatrix():
    """Wrapper for `sklearn.metrics.confusion_matrix`"""

    def __init__(self, y_true, y_pred):
        self.cm_ = sklm.confusion_matrix(y_true, y_pred)

    @property
    def as_dataframe(self):
        """Create pandas.DataFrame and return.
        Names rows and columns."""
        if not hasattr(self, 'df'):
            self.df = pd.DataFrame(self.cm_)
            self.df.index.name = 'true'
            self.df.columns.name = 'pred'
        return self.df

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


def get_label_binary_classification(y_true:int, y_pred:int) -> str:
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


# # for testing
# import numpy as np
# for y_true, y_pred, label in zip(np.array([1, 1, 0, 0]),
#                                  np.array([1, 0, 1, 0]),
#                                  ['TP', 'FN', 'FP', 'TN']):
#     assert get_label_binary_classification(y_true, y_pred) == label


def get_score(clf, X:pd.DataFrame, pos=1) -> pd.Series:
    scores = clf.predict_proba(X)
    if scores.shape[-1] > 2:
        raise NotImplementedError
    else:
        scores = scores[:,pos]
    scores = pd.Series(scores, index=X.index)
    return scores


def get_pred(clf, X:pd.DataFrame) -> pd.Series:
    ret = clf.predict(X)
    ret = pd.Series(ret, index=X.index)
    return ret

def get_custom_pred(clf, X: pd.DataFrame, cutoff=0.5) -> pd.Series:
    scores = get_score(clf, X)
    ret = (scores > cutoff).astype(int)
    return ret
    
    

def get_target_count_per_bin(score: pd.Series, y: pd.Series, n_bins: int = 10):
    pred_bins = pd.DataFrame({
        'score':
        pd.cut(score, bins=list(x / n_bins for x in range(0, n_bins + 1))),
        'y==1':
        y
    })
    pred_bins = pred_bins.groupby(by='score').sum().astype(int)
    return pred_bins