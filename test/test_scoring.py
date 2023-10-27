# # for testing
from njab.sklearn import scoring
import numpy as np


def test_get_label_binary_classification():
    for y_true, y_pred, label in zip(np.array([1, 1, 0, 0]),
                                     np.array([1, 0, 1, 0]),
                                     ['TP', 'FN', 'FP', 'TN']):
        assert scoring.get_label_binary_classification(y_true, y_pred) == label
