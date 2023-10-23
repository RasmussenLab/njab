import pandas as pd
from sklearn import preprocessing


class StandardScaler(preprocessing.StandardScaler):
    """Standardscaler which keeps column names and indices of pandas DataFrames."""

    def transform(self, X, copy=None):
        res = super().transform(X, copy)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(res, columns=X.columns, index=X.index)
        return res

    def inverse_transform(self, X, copy=None):
        res = super().inverse_transform(X, copy)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(res, columns=X.columns, index=X.index)
        return res
