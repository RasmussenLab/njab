import numpy as np
import numpy.testing as npt
import pandas as pd
import sklearn.preprocessing as preprocessing

from njab.sklearn.preprocessing import StandardScaler


def test_StandardScaler():
    X = pd.DataFrame(np.array([[2, None], [3, 2], [4, 6]]))
    npt.assert_almost_equal(preprocessing.StandardScaler().fit(X).transform(X),
                            StandardScaler().fit(X).transform(X).to_numpy())
