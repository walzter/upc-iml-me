from Labs.IBL.algorithms.kibl import kibl, evaluate_performance
import pandas as pd
import numpy as np

def run_kibl():
    dataset_name = 'autos'
    k = 1
    K_IBL_classify = kibl(dataset_name, k)
    print('Dataset =', dataset_name, 'and k =', k)
    print('Accuracy % for (Euclidean, Manhattan, Clark, HVDM) = ', evaluate_performance(K_IBL_classify))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_kibl()
