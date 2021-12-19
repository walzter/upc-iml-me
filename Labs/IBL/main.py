from Labs.IBL.algorithms.kibl import kibl, evaluate_performance

def run_kibl():
    dataset_name = 'autos'
    k = 5
    voting_protocol = 'modified_plurality'
    feature_selection_flag = True
    feature_selection_method = 'rfe'
    no_of_features = 11
    print('Dataset = ' + dataset_name + ' | k = ' + str(k) + ' | Voting protocol = ' + voting_protocol + ' | Feature selection on? = ' + str(feature_selection_flag)+'/'+str(feature_selection_method)+'/'+str(no_of_features)+' features')
    K_IBL_classify, perf_time = kibl(dataset_name, k, voting_protocol, feature_selection_flag, feature_selection_method, no_of_features)
    print('Accuracy % for (Euclidean, Manhattan, Clark, HVDM) = ', evaluate_performance(K_IBL_classify))
    print('Time taken to classify (across all 10 folds) =', perf_time)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_kibl()
