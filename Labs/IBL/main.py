from Labs.IBL.algorithms.kibl import kibl, evaluate_performance
import numpy as np

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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_kibl()
