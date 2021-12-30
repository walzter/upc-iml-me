import numpy as np
import pandas as pd
import numbers


def euclidean_sim(x_pred, x_train):
    def only_numeric(lst):
        only_nums = [x for x in lst if isinstance(x, numbers.Number)]
        only_nums = pd.Series(only_nums)
        return only_nums

    x_pred_num = only_numeric(x_pred)
    x_train_num = only_numeric(x_train)
    x_pred_cat = list(set(x_pred) - set(x_pred_num))
    x_train_cat = list(set(x_train) - set(x_train_num))

    num_diff = x_pred_num - x_train_num if x_pred_num.shape[0] > 0 else []
    cat_diff = [1 if cat_a != cat_b else 0 for cat_a, cat_b in zip(x_pred_cat, x_train_cat)]

    try:
        return np.linalg.norm(num_diff + cat_diff)
    except Exception as e:
        return 0
