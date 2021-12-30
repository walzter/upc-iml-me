import pandas as pd
import time

from Labs.IBL.utils.file_utils import get_test_and_train_data_for_fold
from Labs.IBL.utils.distance_metric_utils import get_distance, append_class_to_dist
from Labs.IBL.utils.voter_utils import voter_most_voted, voter_modified_plurality, voter_borda_count
from Labs.IBL.utils.preprocessing_utils import preprocess_data, select_features, prune_non_essential_features

def kibl(dataset_name, k, voting_protocol, feature_selection_enabled, feature_selection_method, threshold_or_no_of_features):
    '''
    Step 0 - Initialise K_prediction_df as an empty DataFrame
    Step 1 - For every fold, read test & train
      Step 1.1 - Preprocess (i.e. normalise numerical data) for both test & train
      Step 1.2 - For every test example, call get_distance(q, x) - will return a tuple with all 4 distances
        Step 1.2.1 - Pass all 4 distances and get corresponding class labels - will return a df per test example
        Step 1.2.2 - [Andrey] Call voter_<mode>(dist_df, k) to return a (c_euclidean, c_manhattan, c_clark, c_hvdm) per example.
                     Append c_actual column,
                     and append the row to K_prediction_df
      Step 1.3 - You now have a K_prediction_df for 1 fold
    Step 2 - You now have K_prediction_df for all folds. Create 4 confusion matrices fo the 4 distance types
    '''
    start = time.time()
    features_already_selected = False
    selected_features = None
    print('Classifying with K =', k)
    K_prediction_df = pd.DataFrame({'c_euclidean': [], 'c_manhattan': [], 'c_clark': [], 'c_hvdm': []})
    for fold_num in range(0, 10):
        print('Processing fold', fold_num)
        x_train, x_test, meta = get_test_and_train_data_for_fold(dataset_name, fold_num)

        # Normalising numerical data
        x_train = preprocess_data(x_train, meta)
        x_test = preprocess_data(x_test, meta)

        # Optional feature selection
        if feature_selection_enabled:
            if features_already_selected:
                x_train, x_test = prune_non_essential_features(x_train, x_test, selected_features)
            else:
                selected_features = select_features(x_train, feature_selection_method, threshold_or_no_of_features)
                x_train, x_test = prune_non_essential_features(x_train, x_test, selected_features)
                features_already_selected = True

        x_train_wo_labels = x_train.drop(list(x_train)[-1], axis=1)
        x_test_wo_labels = x_test.drop(list(x_test)[-1], axis=1)
        # print('# of samples in train =', len(x_train_wo_labels), ', test =', len(x_test_wo_labels))

        for index, q in x_test_wo_labels.iterrows():
            q_df = pd.DataFrame()
            q_df = pd.concat([q_df, q.to_frame().T])
            e, m, c, h = get_distance(x_train_wo_labels, q_df, meta, feature_selection_enabled, selected_features)
            dist_df = append_class_to_dist(e, m, c, h, x_train)

            if (voting_protocol == 'modified_plurality'):
                final_class_e, final_class_m, final_class_c, final_class_h = voter_modified_plurality(dist_df, k)
            elif (voting_protocol == 'borda_count'):
                final_class_e, final_class_m, final_class_c, final_class_h = voter_borda_count(dist_df, k)
            else:
                # Apply the simplest voting algo of most votes by default
                final_class_e, final_class_m, final_class_c, final_class_h = voter_most_voted(dist_df, k)

            temp_df = pd.DataFrame([[final_class_e, final_class_m, final_class_c, final_class_h, x_test.loc[index].values[-1]]],
                                   columns=['c_euclidean', 'c_manhattan', 'c_clark', 'c_hvdm', 'c_actual'])
            K_prediction_df = pd.concat([temp_df, K_prediction_df])
        # print('Size of pred DF at end of fold', fold_num, ' = ', len(K_prediction_df))
    end = time.time()
    return K_prediction_df, "{:.2f}".format((end - start)) + "s"


def evaluate_performance(K_prediction_df):
    actual_classes = K_prediction_df['c_actual'].values

    correct_count = 0
    for index, c_e in enumerate(K_prediction_df['c_euclidean'].values):
        if c_e == actual_classes[index]:
            correct_count += 1
    e_perf = (correct_count * 100) / len(K_prediction_df)

    correct_count = 0
    for index, m_e in enumerate(K_prediction_df['c_manhattan'].values):
        if m_e == actual_classes[index]:
            correct_count += 1
    m_perf = (correct_count * 100) / len(K_prediction_df)

    correct_count = 0
    for index, c_c in enumerate(K_prediction_df['c_clark'].values):
        if c_c == actual_classes[index]:
            correct_count += 1
    c_perf = (correct_count * 100) / len(K_prediction_df)

    correct_count = 0
    for index, c_h in enumerate(K_prediction_df['c_hvdm'].values):
        if c_h == actual_classes[index]:
            correct_count += 1
    h_perf = (correct_count * 100) / len(K_prediction_df)

    return "{:.2f}".format(e_perf), "{:.2f}".format(m_perf), "{:.2f}".format(c_perf), "{:.2f}".format(h_perf)