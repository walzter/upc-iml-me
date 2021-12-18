import numpy as np
import pandas as pd

np.seterr(divide='ignore', invalid='ignore')


def get_distance(x, q, meta):
    '''
    x = training examples; type DataFrame
    q = query example; type DataFrame
    '''
    numerical_manhattan_dist = np.zeros((len(x)))
    numerical_euclidean_dist = np.zeros((len(x)))
    numerical_clark_dist = np.zeros((len(x)))
    numerical_hvdm_dist = np.zeros((len(x)))
    categorical_dist = np.zeros((len(x)))

    # First, divide each DF to 2 DFs - numerical only and categorical only
    numerical_col_names = [meta.names()[:-1][i] for i in [index for index, type in enumerate(meta.types()[:-1]) if type == 'numeric']]
    categorical_col_names = [meta.names()[:-1][i] for i in [index for index, type in enumerate(meta.types()[:-1]) if type == 'nominal']]
    x_numerical = x.drop(categorical_col_names, axis=1)
    x_categorical = x.drop(numerical_col_names, axis=1)
    q_numerical = q.drop(categorical_col_names, axis=1)
    q_categorical = q.drop(numerical_col_names, axis=1)

    # Checking if there is numerical data
    if (x_numerical.shape[1] != 0):
        # Computing numerical distance
        # TODO handle one or more missing values as:
        # Missing attribute values are assumed to be maximally different from the value present.
        # If they are both missing, then f(xi, Yi) yields 1.

        # For now, ignoring distances for features in which there is missing value in either train or query.
        # "maximally diff" - ?
        x_num_values = x_numerical.values
        q_num_values = q_numerical.values

        missing_val_indices_in_query = np.argwhere(pd.isnull(q_num_values[0]))

        # Calculating all 4 distances
        for row_num in range(0, len(x)):
            missing_val_indices_in_train = np.argwhere(pd.isnull(x_num_values[row_num]))  # indexof missing feature value in that row/exaple
            missing_feature_indices_in_both = [index for index in missing_val_indices_in_train if index in missing_val_indices_in_query]
            for feature_ind in range(0, len(x_num_values[row_num])):
                if feature_ind in missing_feature_indices_in_both:
                    numerical_manhattan_dist[row_num] += 1
                    numerical_euclidean_dist[row_num] += 1
                    numerical_clark_dist[row_num] += 1
                    numerical_hvdm_dist[row_num] += 1
                elif feature_ind not in missing_val_indices_in_train and feature_ind not in missing_val_indices_in_query:
                    numerical_manhattan_dist[row_num] += abs(x_num_values[row_num][feature_ind] - q_num_values[0][feature_ind])
                    numerical_euclidean_dist[row_num] += ((x_num_values[row_num][feature_ind] - q_num_values[0][feature_ind]) ** 2)
                    numerical_clark_dist[row_num] += ((abs(x_num_values[row_num][feature_ind] - q_num_values[0][feature_ind]) ** 2) / (abs(x_num_values[row_num][feature_ind] + q_num_values[0][feature_ind]) ** 2))
                    numerical_hvdm_dist[row_num] += 0 # TODO
            numerical_euclidean_dist[row_num] = np.sqrt(numerical_euclidean_dist[row_num])


    # Checking if there is categorical data
    if (x_categorical.shape[1] != 0):
        # Computing categorical distance
        x_cat_values = x_categorical.values
        q_cat_values = q_categorical.values

        for row_num in range(0, len(x_cat_values)):
            for index, q_cat_val in enumerate(q_cat_values[0]):
                if q_cat_val != x_cat_values[row_num][index]:
                    categorical_dist[row_num] += 1

    return numerical_euclidean_dist + categorical_dist, numerical_manhattan_dist + categorical_dist, numerical_clark_dist + categorical_dist, numerical_hvdm_dist + categorical_dist


def append_class_to_dist(euc_dist_vector, man_dist_vector, clark_dist_vector, hvdm_dist_vector, x):
    '''
    dist_vector = distance of query example to corresponding training examples; type 1D numpy array
    x = training examples; type DataFrame
    '''
    dist_class_df = pd.DataFrame({'euclidean_distance': euc_dist_vector, 'manhattan_distance': man_dist_vector,
                                  'clark_distance': clark_dist_vector, 'hvdm_distance': hvdm_dist_vector,
                                  'class': x[list(x)[-1]].values})
    return dist_class_df