import numpy as np
import pandas as pd


def euclidean_sim(x_pred, x_train):
    # TODO Split data between numerical and categorical
    # Apply xp[i] != xt[i] so 0 if equals, 1 otherwise
    # Apply euclidean for numerical + categorical
    # Possible options would be:
    #   - Check each element type in the array and perform op
    #   - Split ds in numerical and categorical, this improves perf
    #   - Apply onehot encoding
    x_pred_num = x_pred._get_numeric_data()
    x_train_num = x_train._get_numeric_data()
    x_pred_cat = list(set(x_pred) - set(x_pred_num))
    x_train_cat = list(set(x_train) - set(x_train_num))

    # Missing how to deal with missing data
    x_pred_miss = np.where(x_pred.isna())
    x_train_miss = np.where(x_train.isna())

    num_diff = x_pred_num - x_train_num if x_pred_num.shape[0] > 0 else []
    cat_diff = [1 if cat_a != cat_b else 0 for cat_a, cat_b in zip(x_pred_cat, x_train_cat)]

    return np.linalg.norm(num_diff + cat_diff)
