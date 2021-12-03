import numpy as np

'''
    Returns the most requent class label
'''


def most_voted_solution(y_pred):
    lst, freq = np.unique(y_pred, return_counts=True)
    votes = np.c_[lst, freq]
    sorted_votes = np.argsort(votes[:, 1])[::-1]

    # Naive selection of the first vote. In case of tie always picks the first in the list, does not care of ties
    return sorted_votes[0]


def modified_plurality(y_pred):
    return


def borda_count(y_pred):
    return
