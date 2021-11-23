import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
from sklearn import metrics

import preprocessing

M = 2

def get_optimal_parameters(dataset):
    if dataset == 'kropt':
        return 8
    if dataset == 'satimage':
        return 3
    if dataset == 'hypothyroid':
        return 2

def find_centroids(memberships, x, c):

    centroids = [sum((memberships.transpose()[i]**M)*x.transpose()) /
                 sum(memberships.transpose()[i] ** M) for i in range(c)]

    return centroids

def update_memberships(distances):
    # TODO: implement equation slides
    memberships = []
    for dist in distances:
        mb_point = []
        for i in range(len(dist)):
            num = np.full((1, len(dist)), dist[i])
            num = [n ** 2 for n in num]
            den = [n ** 2 for n in dist]
            memb = 1 / pow(sum(np.divide(num, den)[0]), 1 / (M - 1))
            mb_point.append(memb)
        memberships.append(mb_point)
    return memberships


def fuzzycmeans(x, c, iterations):
    # Randomly initialize membership of each point to clusters
    membership_init = []
    for _ in range(len(x)):
        a = np.random.random(c)
        a /= sum(a)
        membership_init.append(a)

    # Find out the centroids
    centroids = find_centroids(np.asarray(membership_init), x, c)

    # Find out the distance of each point from centroid
    # Array of arrays: each array represents a point and inside it the distances to every centroid are stored
    distances = euclidean_distances(x, centroids)


    # Updating membership values
    memberships = update_memberships(distances)

    # Repeat the above steps for a defined number of iterations
    for _ in tqdm(range(iterations)):
        # Find out the centroids
        centroids = find_centroids(memberships, x, c)

        # Find out the distance of each point from centroid
        # Array of arrays: each array represents a point and inside it the distances to every centroid are stored
        distances = euclidean_distances(x, centroids)

        # Centroid with the minimum Distance
        memberships = update_memberships(distances)

    memberships = np.argmax(memberships, axis=1)

    return memberships

def silhouette_plot(traindata):
    total_runs = []
    for _ in range(5):
        performance = []
        C = range(2, 11)
        for c in C:
            data = traindata.copy()
            prediction = fuzzycmeans(data, c, 50)
            performance.append(round(metrics.silhouette_score(traindata, prediction), 4))
        total_runs.append(performance)
    avg = np.average(np.asarray(total_runs), axis=0)
    plt.plot(C, avg, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.show()


def fuzzycmeans_algorithm(traindata, dataset, task):
    if task == 'tune':
        silhouette_plot(traindata)
    elif task == 'results':
        print(dataset)
        c = get_optimal_parameters(dataset)
        prediction = fuzzycmeans(traindata, c, 10)

        # Calculate and print silhouette score
        val = round(metrics.silhouette_score(traindata, prediction), 4)
        print('Silhouette coefficient: ' + str(val))

        # Visualize the results
        cluster_membership = np.argmax(prediction, axis=1)
        two_dim_data = preprocessing.principal_component_analysis(traindata)
        plt.scatter(two_dim_data.iloc[:, 0], two_dim_data.iloc[:, 1], c=cluster_membership, s=1)
        plt.show()
    else:
        print('Enter a valid task')