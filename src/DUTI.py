"""
The original algorithm: DUTI
Created by shouxing, 2019/6/27
"""
import numpy as np


class DUTI:
    """
    Correct the labels of the training data items using the labels of the trusted items.
    """
    def __init__(self, num_class, lam=1, max_iter=20):
        """

        :param num_class:
        :param lam: ridge coefficient of the learner, positive real number
        :param max_iter: maximum iteration
        """
        self.num_class = num_class
        self.lam = lam
        self.max_iter = max_iter

    def fit_transform(self, feature, label, feature_trusted, label_trusted, confidence=None):
        """

        :param feature:
        :param label:
        :param feature_trusted:
        :param label_trusted:
        :param conf: confidence vector of trusted items, m x 1 vector
        :return:
        """
        if confidence is None:
            self.confidence = np.ones_like(label_trusted)
        else:
            self.confidence = confidence
        pass