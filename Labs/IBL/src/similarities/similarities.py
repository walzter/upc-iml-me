import numpy as np


def euclidean_sim(x_pred, x_train):
    # TODO Split data between numerical and categorical
    # Apply xp[i] != xt[i] so 0 if equals, 1 otherwise
    # Apply euclidean for numerical + categorical
    # Possible options would be:
    #   - Check each element type in the array and perform op
    #   - Split ds in numerical and categorical, this improves perf
    #   - Apply onehot encoding

    return np.linalg.norm(x_pred - x_train)
