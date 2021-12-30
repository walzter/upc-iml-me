import numpy as np

from src.algorithms.i_ibl import IIBL


class IBL2(IIBL):
    def __init__(self, sim_func):
        self.sim_func = sim_func
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        CD = [x_train.iloc[0]]
        CD_Y = [y_train[0]]

        for i in range(0, len(x_train)):
            cd_pred = CD_Y[self._best_local(CD, x_train.iloc[i])]

            ## If not found in CD, add it as a sample
            if not cd_pred == y_train[i]:
                CD_Y.append(y_train[i])
                CD.append(x_train.iloc[i])

        self.x_train = CD
        self.y_train = CD_Y

    def _best_local(self, sample_pool, to_match):
        sim = np.zeros(len(sample_pool))

        for i in range(0, len(sample_pool)):
            sim[i] = self.sim_func(to_match, sample_pool[i])

        return np.argmin(sim)

    def predict(self, x_pred):
        y_index = self._best_local(self.x_train, x_pred)

        return self.y_train[y_index]
