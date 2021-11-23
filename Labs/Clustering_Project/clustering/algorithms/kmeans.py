import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import preprocessing
from sklearn.metrics import silhouette_score, silhouette_samples, confusion_matrix, ConfusionMatrixDisplay, \
    multilabel_confusion_matrix, precision_score, recall_score, f1_score
from io import BytesIO

from sklearn.preprocessing import MultiLabelBinarizer

from utils.file_reader import load_arff_file

DATA_SETS = [{
        'name': 'kropt',
        'file_path': './datasets/kropt.arff',
        'k_cluster_sizes': [2, 3, 4, 5, 8]
    },
    {
        'name': 'soybean',
        'file_path': './datasets/soybean.arff',
        'k_cluster_sizes': [20, 17, 19]
    },
    {
        'name': 'iris',
        'file_path': './datasets/iris.arff',
        'k_cluster_sizes': [4, 3, 2]
    }]

class Kmeans:
    def __init__(self, k_clusters=3, max_iter=100):
        self.k_clusters = k_clusters
        self.max_iter = max_iter
        self.centroids = {}
        self.prev_centroids = []
        self.nearest_cluster = []

    def init_centroids(self, data):
        initial_indexes = np.random.permutation(data.shape[0])[:self.k_clusters]
        centroids = data[initial_indexes]

        return centroids

    def calculate_centroids(self, data, nearest_cluster):
        centroids = np.zeros((self.k_clusters, data.shape[1]))

        for k in range(self.k_clusters):
            centroids[k, :] = np.mean(data[nearest_cluster == k, :], axis=0)
        return centroids

    def get_nearest_neighbor(self, data):
        return np.argmin(data, axis=1)

    def fit(self, data):
        self.centroids = self.init_centroids(data)

        for i in range(self.max_iter):
            self.prev_centroids = self.centroids
            all_distances = self.calculate_distance(data, self.prev_centroids)
            self.nearest_cluster = self.get_nearest_neighbor(all_distances)
            self.centroids = self.calculate_centroids(data, self.nearest_cluster)

            if np.all(self.centroids == self.prev_centroids):
                break

    def calculate_distance(self, data, centroids):
        euclidean_dist = np.zeros((data.shape[0], self.k_clusters))

        for i in range(self.k_clusters):
            distance_values = np.linalg.norm(data - centroids[i], axis=1)
            euclidean_dist[:, i] = np.square(distance_values)

        return euclidean_dist

    def predict(self, sample):
        distance = self.calculate_distance(sample, self.centroids)
        return self.get_nearest_neighbor(distance)


def perform_kmeans():
    names = map(lambda dataset: dataset.get('name'), DATA_SETS)
    user_input = -1

    while user_input != 0:
        print('The available dataset names are:\n')
        [print(f'{i+1} - {name}\n') for i, name in enumerate(names)]
        print('0 - Exit')
        user_input = int(input('Please select type the number related to the action you want to perform.\n ->'))
        if user_input == 0 or user_input > len(DATA_SETS):
            continue

        dataset = DATA_SETS[user_input-1]
        data_array, data_array_with_classes, _ = load_arff_file(dataset.get('file_path'))
        mlb = MultiLabelBinarizer()
        mlb.fit_transform([(1, 2), (3,)])
        perform_analysis(data_array, data_array_with_classes[:, -1], dataset.get('k_cluster_sizes'), dataset.get('name'))

def compute_confusion(y_train, y_predict, k_clusters, title):
    conf_m = multilabel_confusion_matrix(y_train, y_predict)

    plt.title(f'Confusion matrix for {k_clusters} clusters')
    for index, entry in enumerate(conf_m):
        ax = plt.subplot(2, 2, index+1)
        disp = ConfusionMatrixDisplay(confusion_matrix=entry)
        disp.plot(ax=ax)
        ax.set_title(f'C={index}')
        ax.margins(2, 2)
    recall = recall_score(y_train, y_predict, average=None)
    precision = precision_score(y_train, y_predict, average=None)
    f1_scr = f1_score(y_train, y_predict, average=None)
    print(f'Recall score per cluster is: {recall}')
    print(f'Precision score  per cluster is: {precision}')
    print(f'F1 score  per cluster is: {f1_scr}')
    plt.tight_layout()
    plt.savefig(fname=f'./plots/{title}_kmeans_conf_{k_clusters}.png')
    plt.show()

    return

def perform_analysis(data_array, defined_labels, k_clusters_sizes, title):
    silhouette_avg = np.zeros_like(k_clusters_sizes, dtype=float)
    le = preprocessing.LabelEncoder()
    proccessed_labels = le.fit_transform(defined_labels)
    fig, ax = plt.subplots(len(k_clusters_sizes), figsize=(12, 5), sharex=True, sharey=True)
    predicted_labels = []
    for index, k_clusters in enumerate(k_clusters_sizes):
        kmeans = Kmeans(k_clusters)
        kmeans.fit(data_array)
        labels = kmeans.predict(data_array)

        predicted_labels.append({
            'clusters': k_clusters,
            'labels': labels
        })

        silhouette_avg[index] = silhouette_score(data_array, labels)

        sample_silhouette_values = silhouette_samples(data_array, labels)

        ax[index].set_xlim([-0.1, 1])
        ax[index].set_ylim([0, len(data_array) + (k_clusters + 1) * 10])

        y_lower = 10
        for i in range(k_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / k_clusters)
            ax[index].fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            ax[index].text(-0.09, y_lower + 0.5 * size_cluster_i, str(i))

            y_lower = y_upper + len(k_clusters_sizes)

        ax[index].set_title(f'Clusters: {k_clusters}')

        ax[index].axvline(x=silhouette_avg[index], color="red", linestyle="--")

        ax[index].set_yticks([])
        ax[index].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    ax[0].set_xlabel("The silhouette coefficient values")
    ax[0].set_ylabel("Cluster label")
    plt.suptitle(
        f'Silhouette analysis for {title}.arff file',
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(fname=f'./plots/{title}_kmeans_values.png')
    plt.show()

    plt.plot(k_clusters_sizes, silhouette_avg)
    plt.suptitle(
        f'Silhouette analysis for {title}.arff file',
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel(r'Number of clusters *k*')
    plt.ylabel('Sum of squared distance')
    plt.savefig(fname=f'./plots/{title}_kmeans_score.png')
    plt.show()

    for pl in predicted_labels:
        compute_confusion(proccessed_labels, pl.get('labels'), pl.get('clusters'), title)
    return
