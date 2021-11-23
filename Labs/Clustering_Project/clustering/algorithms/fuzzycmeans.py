import numpy as np
from matplotlib import pyplot as plt, cm
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler, Normalizer, normalize

from utils.file_reader import load_arff_file

DATA_SETS = [{
    'name': 'wine',
    'file_path': './datasets/wine.arff',
    'k_cluster_sizes': [2, 3, 4, 5, 8]
}]


class FuzzyCmeans:
    FUZZINESS_DEGREE = 2

    def __init__(self, k_clusters=3, max_iter=100):
        # n is number of rows in X
        # X = { x1, x2, ..., xn}, x[i] e R^p, p = number of features in each vector
        # OUTPUT:
        #   matrix[c, U], c partition of X where c=number of clusters and U=universe
        #   vectors V = {v1, v2, ..., vc} c R^p
        #    vi is a cluster center

        self.max_iter = max_iter
        self.k_clusters = k_clusters
        self.centroids = {}
        self.clusters = []
        self.prev_centroids = []
        self.membership_matrix = []
        self.fuzz_exp = 2/(self.FUZZINESS_DEGREE-1)

        return

    def init_centroids(self, data):
        initial_indexes = np.random.permutation(data.shape[0])[:self.k_clusters]

        centroids = data[initial_indexes]

        return centroids

    def calculate_membership1(self, data, centroids):
        membership_matrix = np.zeros((data.shape[0], self.k_clusters))
        euclidean_dist = np.zeros((data.shape[0], self.k_clusters))

        for i in range(self.k_clusters):
            euclidean_dist[:, i] = np.linalg.norm(data - centroids[i], axis=1)

        for i in range(self.k_clusters):
            numerator = euclidean_dist[:, i]

            for j in range(self.k_clusters):
                denominator = euclidean_dist[:, j]
                membership = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
                membership = pow(membership, self.fuzz_exp)

                membership_matrix[:, i] = membership_matrix[:, i] + np.divide(1, membership, out=np.zeros_like(membership), where=membership != 0)

        return membership_matrix

    def calculate_membership(self, data, centroids):
        membership_matrix = np.zeros((data.shape[0], self.k_clusters))
        euclidean_dist = np.zeros((data.shape[0], self.k_clusters))

        for i in range(self.k_clusters):
            euclidean_dist[:, i] = np.linalg.norm(data - centroids[i], axis=1)

        for i in range(self.k_clusters):
            numerator = euclidean_dist[:, i]

            for j in range(self.k_clusters):
                denominator = euclidean_dist[:, j]
                membership = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
                membership = pow(membership, self.fuzz_exp)

                membership_matrix[:, i] = membership_matrix[:, i] + np.divide(1, membership,
                                                                              out=np.zeros_like(membership),
                                                                              where=membership != 0)

        return membership_matrix

    def calculate_centroids(self, membership_functions, data):
        centroids = np.zeros((self.k_clusters, data.shape[1]))
        sqr_memberships = membership_functions**self.FUZZINESS_DEGREE

        for i in range(self.k_clusters):
            numerator = np.sum([sqr_memberships[k, i] * data[k] for k in range(data.shape[0])], axis=0)
            denominator = np.sum(sqr_memberships[:, i])

            centroids[i] = numerator/denominator

        return centroids

    def fit(self, data):
        self.membership_matrix = np.zeros((data.shape[0], self.k_clusters))
        self.centroids = self.init_centroids(data)

        for i in range(self.max_iter):
            self.prev_centroids = self.centroids
            self.membership_matrix = self.calculate_membership(data, self.centroids)
            self.centroids = self.calculate_centroids(self.membership_matrix, data)

            if np.all(self.centroids == self.prev_centroids):
                break

        return self.membership_matrix, self.centroids

    # Used to graph the silhouette values
    def get_nearest_neighbor(self, data):
        return np.argmin(data, axis=1)

    # Returned 'distance' as the membership function values
    # Returned 'nearest neighbor' as a crisp set to perform silhouette evaluations
    def predict(self, sample):
        distance = self.calculate_membership(sample, self.centroids)
        return distance, self.get_nearest_neighbor(distance)


def perform_fuzzycmeans():

    user_input = -1

    while user_input != 0:
        print('The available dataset names are:\n')
        names = map(lambda dataset: dataset.get('name'), DATA_SETS)
        for i, name in enumerate(names):
            print(f'{i + 1} - {name}\n')
        print('0 - Exit')
        user_input = int(input('Please select type the number related to the action you want to perform.\n ->'))
        if user_input == 0 or user_input > len(DATA_SETS):
            continue

        dataset = DATA_SETS[user_input-1]
        data_array, data_array_with_classes, _ = load_arff_file(dataset.get('file_path'))
        data_array = np.copy(data_array)
        scaler = StandardScaler()
        data_array = scaler.fit_transform(data_array)
        transformer = Normalizer().fit(data_array)
        data_array = transformer.transform(data_array)

        perform_analysis(data_array, dataset.get('k_cluster_sizes'), dataset.get('name'))


def perform_analysis(data_array, k_clusters_sizes, title):
    fig, ax = plt.subplots(1, len(k_clusters_sizes), figsize=(12, 5))

    ax[0].set_xlabel("The silhouette coefficient values")
    ax[0].set_ylabel("Cluster height")
    for index, k_clusters in enumerate(k_clusters_sizes):

        fuzzycmeans = FuzzyCmeans(k_clusters)
        u_matrix, centroids = fuzzycmeans.fit(data_array)
        fuzzy_distances, labels = fuzzycmeans.predict(data_array)
        fd_mean = normalize(fuzzy_distances)
        #fd_mean = np.median(fd_mean, axis=0)
        for i in range(k_clusters):
            color = cm.nipy_spectral(float(i) / k_clusters)

            ax[index].plot(
                range(fd_mean.shape[0]),
                fd_mean[:, i],
                color=color
            )

        ax[index].set_title(f'Clusters: {k_clusters}')

    plt.show()

    return
