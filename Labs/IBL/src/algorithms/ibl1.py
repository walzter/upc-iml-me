import numpy as np

from src.algorithms.i_ibl import IIBL


class IBL1(IIBL):
    def __init__(self, sim_func):
        self.sim_func = sim_func
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

        return

    def predict(self, x_pred):
        sim = []

        for i, entry in enumerate(self.x_train):
            sim[i] = self.sim_func(x_pred, entry)

        y_pred = self.y_train[np.argmin(sim)]

        self.x_train.append(x_pred)
        self.y_train.append(y_pred)

        return y_pred
