import numpy as np
from scipy import stats
from utils.categoric_dissimilarity_calculator import get_dissim_categoric
from utils.file_reader import load_arff_file
import pandas as pd

import plotly.express as px
from utils.common import list_to_numbers

DATA_SETS = [
    {
        'name': 'soybean',
        'file_path': './datasets/soybean.arff',
        'k_cluster_sizes': [20, 17, 19],
        'lbl_column': 'class'
    },
    {
        'name': 'kropt',
        'file_path': './datasets/kropt.arff',
        'k_cluster_sizes': [2, 3, 4, 5],
        'lbl_column': 'game'
    },
    {
        'name': 'iris',
        'file_path': './datasets/iris.arff',
        'k_cluster_sizes': [4, 3, 2]
    }]

def perform_kmodes():
    user_input = -1

    while user_input != 0:
        print('The available dataset names are:\n')
        names = map(lambda dataset: dataset.get('name'), DATA_SETS)
        [print(f'{i+1} - {name}\n') for i, name in enumerate(names)]
        print('0 - Exit')
        user_input = int(input('Please select type the number related to the action you want to perform.\n ->'))
        if user_input == 0 or user_input > len(DATA_SETS):
            continue

        dataset = DATA_SETS[user_input-1]
        data_array, data_array_with_classes, type_metadata = load_arff_file(dataset.get('file_path'), dataset.get('lbl_column'))

        if type_metadata.types()[:-1].count('nominal') == 0:
            print('Chosen dataset does not contain any categorical data, aborting run of K-Modes algorithm')
            return
        perform_analysis(data_array, type_metadata, data_array_with_classes, dataset.get('k_cluster_sizes'), dataset.get('name'))

def perform_analysis(data_array, type_metadata, data_array_with_classes, k_clusters_sizes, title):
    actual_classes = data_array_with_classes[:, -1]
    for index, k_clusters in enumerate(k_clusters_sizes):
        kmodes = Kmodes(k_clusters)
        kmodes.fit(data_array, type_metadata)
        predicted_labels = kmodes.predict(data_array, type_metadata)

        df = pd.DataFrame({'Labels': actual_classes, 'Clusters': predicted_labels})
        # Create crosstab matrix ct; with labels and features as columns
        ct = pd.crosstab(df['Labels'], df['Clusters'])
        print('Crosstab matrix for ', title, 'dataset & [', k_clusters, ' clusters]:\n', ct)
    # Only visualising (for now) for the best value of "k" - which is the last element in k_clusters_size list for every dataset
    visualise(data_array_with_classes, predicted_labels, type_metadata)

class Kmodes:
    def __init__(self, k_clusters=2, max_iter=5):
        # Initialising default values
        self.k_clusters = k_clusters
        self.max_iter = max_iter
        self.data_size = 0
        self.feature_size = 0

        # Initialising empty clusters
        self.clusters_as_data = list()
        self.clusters_as_indices = list()
        self.cc_index = list()
        self.cc = list()

    def fit(self, data_array, type_metadata):
        self.data_size = data_array.shape[0]
        self.feature_size = len(data_array[0])

        # Initialising dissimilarity matrix as an all-zero matrix
        dissimilarity = np.zeros((self.data_size, self.k_clusters))

        # Get k random rows as seeds
        # Assign the newly chosen cluster centres into the cluster lists
        self.cc_index.extend(np.random.permutation(data_array.shape[0])[:self.k_clusters])
        self.cc.extend(data_array[self.cc_index])
        self.clusters_as_data.extend(data_array[self.cc_index])
        self.clusters_as_indices.extend(self.cc_index)
        print('Initial cluster centers: ', self.cc_index)

        for iteration in range(0, self.max_iter):
            print('[', self.k_clusters, 'clusters] Running iteration #', iteration)

            # For each of the rows, get the dissimilarity matrix D = data_size x K
            for data_no in range(0, self.data_size):
                # For one particular row
                for cluster_no in range(0, self.k_clusters):
                    # Dissim with jth cluster
                    dissimilarity[data_no][cluster_no] = get_dissim_categoric(data_array[data_no], self.cc[cluster_no],
                                                                              self.feature_size, type_metadata)

            # The min of the dissimilarity measure for that data point, gets merged with the corresponding cluster
            # Hence you get a new bigger cluster Resetting clusters before reassignment
            self.clusters_as_data = [[] for cluster in self.clusters_as_data]
            self.clusters_as_indices = [[] for cluster in self.clusters_as_indices]

            for data_no in range(0, self.data_size):
                # For one particular data row
                cc_to_be_assigned = np.argmin(dissimilarity[data_no])
                self.clusters_as_data[cc_to_be_assigned].append(data_array[data_no])
                self.clusters_as_indices[cc_to_be_assigned].append(data_no)

            # Recompute the cluster centres for all clusters by mode of that feature
            for cluster_no in range(0, self.k_clusters):
                for feature_no in range(0, self.feature_size):
                    fth_feature_sublist = []
                    for d in range(0, len(self.clusters_as_data[cluster_no])):
                        fth_feature_sublist.append(self.clusters_as_data[cluster_no][d][feature_no])
                    feature_mode = stats.mode(fth_feature_sublist)[0]
                    self.cc[cluster_no][feature_no] = feature_mode[0]

    def predict(self, test_data, type_metadata):
        predicted_labels = []
        test_dissimilarity = np.zeros((len(test_data), self.k_clusters))
        for data_no in range(0, len(test_data)):
            # For one particular row
            for cluster_no in range(0, self.k_clusters):
                # Dissim with jth cluster
                test_dissimilarity[data_no][cluster_no] = get_dissim_categoric(test_data[data_no], self.cc[cluster_no],
                                                                               self.feature_size, type_metadata)
        for data_no in range(0, len(test_data)):
            # For one particular data row
            predicted_labels.append(np.argmin(test_dissimilarity[data_no]))
        return predicted_labels

def visualise(data_array_with_classes, predicted_labels, type_metadata):
    categorical_colnames = []
    for index, col in enumerate(type_metadata.names()):
        if type_metadata.types()[index] == 'nominal':
            categorical_colnames.append(col)
    categorical_colnames = categorical_colnames[:-1]
    df = pd.DataFrame(data_array_with_classes, columns = type_metadata.names())
    str_df = df.stack().str.decode('utf-8').unstack()

    str_df.insert(1, "predclass", list_to_numbers(set(predicted_labels), predicted_labels), True)
    fig = px.parallel_categories(str_df, categorical_colnames, color="predclass", color_continuous_scale=px.colors.sequential.Inferno)
    fig.show()
