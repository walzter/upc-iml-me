import pandas as pd

from algorithms.kibl import kibl, evaluate_performance
import numpy as np
from src.algorithms.ibl1 import IBL1
from src.algorithms.ibl2 import IBL2
from src.algorithms.ibl3 import IBL3
from src.similarities.similarities import euclidean_sim
from src.util.cross_val import cross_val
from util.file_loader import load_dataset, load_datasets

#sys.path.append('./datasetsCBR')

def something():
    print('Hi skal')

def ibl1(filename):
    algorithm = 'ilb1'
    ds_list = load_datasets(filename)
    performance_results, execution_time = cross_val(ds_list, IBL1(euclidean_sim), 'fit', 'predict')
    df = pd.DataFrame(performance_results)
    df.to_csv(f'./logs/{filename}.{algorithm}.performance.csv')
    df = pd.DataFrame(execution_time)
    df.to_csv(f'./logs/{filename}.{algorithm}.times.csv')

def ibl2(filename):
    algorithm = 'ilb2'
    ds_list = load_datasets(filename)
    performance_results, execution_time = cross_val(ds_list, IBL2(euclidean_sim), 'fit', 'predict')
    df = pd.DataFrame(performance_results)
    df.to_csv(f'./logs/{filename}.{algorithm}.performance.csv')
    df = pd.DataFrame(execution_time)
    df.to_csv(f'./logs/{filename}.{algorithm}.times.csv')

def ibl3(filename):
    algorithm = 'ilb3'
    ds_list = load_datasets(filename)
    performance_results, execution_time = cross_val(ds_list, IBL3(euclidean_sim), 'fit', 'predict')
    df = pd.DataFrame(performance_results)
    df.to_csv(f'./logs/{filename}.{algorithm}.performance.csv')
    df = pd.DataFrame(execution_time)
    df.to_csv(f'./logs/{filename}.{algorithm}.times.csv')

def iblflow():
    # used soybean, credit-a, satimage
    filename = 'soybean'
    ibl1(filename)
    ibl2(filename)
    ibl3(filename)

def run_kibl():
    dataset_name = 'soybean'
    voting_protocol = 'modified_plurality'
    feature_selection_flag = True
    feature_selection_method = 'rfe'
    threshold_or_no_of_features = 17

    k_accuracy = [[] for i in range(1, 5)]
    k_time_taken = np.zeros((9))
    for index, k in enumerate([1, 3, 5, 7]):
        print('Dataset = ' + dataset_name + ' | k = ' + str(k) + ' | Voting protocol = ' + voting_protocol + ' | Feature selection on? = ' + str(feature_selection_flag) + '/' + str(feature_selection_method) + '/' + str(threshold_or_no_of_features) + ' threshold/features')
        K_IBL_classify, perf_time = kibl(dataset_name, k, voting_protocol, feature_selection_flag, feature_selection_method, threshold_or_no_of_features)
        k_accuracy[index] = evaluate_performance(K_IBL_classify)
        print('Accuracy % for (Euclidean, Manhattan, Clark, HVDM) =', k_accuracy[k-1])
        k_time_taken[index] = perf_time
        print('Time taken to classify (across all 10 folds) =', perf_time)

if __name__ == '__main__':
    iblflow()
    #run_kibl()
