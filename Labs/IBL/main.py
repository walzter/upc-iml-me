from Labs.IBL.algorithms.kibl import kibl, evaluate_performance

def run_kibl():
    dataset_name = 'autos'
    k = 5
    voting_protocol = 'modified_plurality'
    K_IBL_classify, perf_time = kibl(dataset_name, k, voting_protocol)
    print('Dataset =', dataset_name, '| k =', k, '| voting protocol =', voting_protocol)
    print('Accuracy % for (Euclidean, Manhattan, Clark, HVDM) = ', evaluate_performance(K_IBL_classify))
    print('Time taken to classify (across all 10 folds) =', perf_time)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_kibl()
