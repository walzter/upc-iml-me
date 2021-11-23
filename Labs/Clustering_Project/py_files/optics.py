import sklearn
from sklearn.cluster import OPTICS,cluster_optics_dbscan
from sklearn.preprocessing import normalize
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys 
sys.path.append('./algorithms')
from helper_funcs import get_le_df
from helper_funcs import *


SAMPLE_RANGES = list(range(10,26))
METRICS = ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan","minkowski"]
ALGS = ['ball_tree','kd_tree','brute']


def clean_optics_cluster(SCALED_DF,Y_true,MIN_SAMPLES,METRIC,ALG,DF,graph='n'):
	optic_params = cluster_dict(MIN_SAMPLES,METRIC,ALG)
	cluster,labels,optic_dict = optic_cluster(SCALED_DF,Y_true, optic_params,DF,graph=graph)
	return SCALED_DF, cluster, labels, optic_dict

def cluster_dict(MIN_SAMPLES,METRIC,ALG):
	params_dict = {
	                "min_samples":MIN_SAMPLES, #points nearby to be considered dense
	                 "metric":METRIC, #list of metrics to use
	                 "algorithm":ALG,
	                 }
	return params_dict

def plot_clusters(normalized_dataframe,clustering,labels,params_dict=None):
    unique_labels = set(labels)
    #clustering.labels = clustering.labels[clustering.labels<2500]
    core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
    core_samples_mask = core_samples_mask[:2500]
    clustering.cluster_hierarchy_ = clustering.cluster_hierarchy_[:2500]
    try: 
        core_samples_mask[clustering.cluster_hierarchy_][:2500] = True
    except Exception as e: 
        pass
    space = np.arange(len(normalized_dataframe))
    space = space[:2500]
    reachability = clustering.reachability_[clustering.ordering_]
    reachability = reachability[:2500]
    #print(reachability)
    labels = clustering.labels_[clustering.ordering_]
    plt.figure(figsize=(10, 7.5))
    colors = ["g.", "r.", "b.", "y.", "c."]
    try:
        for klass, color in zip(range(0, 5), colors):
            Xk = space[labels == klass]
            Rk = reachability[labels == klass]
            plt.plot(Xk, Rk, color, alpha=0.5)
            plt.plot(space[labels == -1], reachability[labels == -1], "k.", alpha=0.3)
            plt.plot(space, np.full_like(space, 2.0, dtype=float), "k-", alpha=0.5)
            plt.plot(space, np.full_like(space, 0.5, dtype=float), "k-.", alpha=0.5)
            plt.ylabel("Reachability (epsilon distance)")
            plt.title("Graph 1. Reachability Plot")
    except Exception as e: 
        pass
    plt.savefig('Graph1_Optics_Reachability.png')
    plt.tight_layout()
    plt.show()
    # OPTICS
    plt.figure(figsize=(10, 7.5))
    colors = ["g.", "r.", "b.", "y.", "c."]
    for klass, color in zip(range(0, 5), colors):
        Xk = normalized_dataframe[clustering.labels_ == klass]
        plt.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
        plt.plot(normalized_dataframe[clustering.labels_ == -1, 0], normalized_dataframe[clustering.labels_ == -1, 1], "k+", alpha=0.1)
        plt.title("Automatic Clustering\nOPTICS")
    #plt.title(f"Graph 1. OPTICS With Clusters:: {params_dict['Clusters']}")
    plt.savefig('Graph2_Optics_Clustering.png')
    plt.tight_layout()
    plt.show()

def optic_cluster(normalized_dataframe,Y_true,params_dict,DF,graph='y'):

	# Clustering starts
    clustering = OPTICS(min_samples=params_dict['min_samples'],
                        metric=params_dict['metric'],
                        algorithm=params_dict['algorithm'])
    #fitting the model
    clustering = clustering.fit(normalized_dataframe)
    #getting the data out
    labels_ordered = clustering.labels_[clustering.ordering_]
    core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
    core_samples_mask[clustering.cluster_hierarchy_] = True
    labels = clustering.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    #df_cat_le = get_le_df(DF, 'class')
    labels_true = (Y_true.values).reshape(-1,)
    #passing to df
    test_df = pd.DataFrame(normalized_dataframe)
    test_df['labels'] = clustering.labels_
    
    d = {
		    "Clusters":n_clusters_,
		    "Noise":n_noise_,
		    "Homogeneity":metrics.homogeneity_score(labels_true, labels),
		    "Completeness": metrics.completeness_score(labels_true, labels),
		    "V-measure": metrics.v_measure_score(labels_true, labels),
		    "Adjusted Rand Index": metrics.adjusted_rand_score(labels_true, labels),
		    "Adjusted Mutual Information":metrics.adjusted_mutual_info_score(labels_true, labels),
		    "Silhouette Coefficient": metrics.silhouette_score(test_df, labels),
    	}
    #plotting the clusters
    if graph =='y':
        plot_clusters(normalized_dataframe,clustering,labels,d)
        return clustering, labels,d
    else:
        return clustering, labels, d

#test the optics with all the metrics and algorithm combinations, we keep num samples the same
# 7 metrics and 3 algorithms 7x3 graphs 
#test the optics with all the metrics and algorithm combinations, we keep num samples the same
# 7 metrics and 3 algorithms 7x3 graphs 
def optics_algs_met_make_gif(X,Y_true,FULL_PATH=None,predefined='y'):
    df = load_dataset(FULL_PATH)
    import itertools
    import sklearn
    import glob
    from PIL import Image
    #Algorithms to be used 
    ALGS = ['ball_tree','kd_tree','brute']
    #metrics which are available in all three:
    a = sorted(sklearn.neighbors.VALID_METRICS['ball_tree'])
    b = sorted(sklearn.neighbors.VALID_METRICS['kd_tree'])
    c = sorted(sklearn.neighbors.VALID_METRICS['brute'])
    METRICS = list(set(a) & set(b) & set(c))
    #METRICS_TO_USE = list(np.random.choice(METRICS, 3,replace=False))
    METRICS_TO_USE = ['minkowski','chebyshev','manhattan']
    print(METRICS_TO_USE)
    #all the coordinates
    if predefined=='y':
        colrange=list(range(0,3))
        rowrange=list(range(0,3))
        COORDS = list(itertools.product(colrange,rowrange))
        alg_met = list(itertools.product(ALGS, METRICS_TO_USE))
    else: 
        colrange = list(range(0,3))
        rowrange = list(range(0,7))
        COORDS = list(itertools.product(colrange,rowrange))
        #all combinations of them
        alg_met = list(itertools.product(ALGS, METRICS))
    # testing all the combinations 
    MIN_SAMPLES = 20
    #creationg the subplot
    fig, ax = plt.subplots(len(rowrange),len(colrange),figsize=(40,20),sharey=True,sharex=True)
    #suptitle
    plt.suptitle("Graph 2 . Clustering: Algorithm and Metric Variations",fontsize=30)
    #storing info
    d = []
    #grid for title above groups
    grid = plt.GridSpec(3, 3)
    for idxx,alg in enumerate(ALGS):
            fake = fig.add_subplot(grid[idxx])
            fake.set_title(f"{alg.upper()}\n", fontweight='semibold', size=14)
            fake.set_axis_off()
    
    #starting to plot all the combinations
    for idx,combs in enumerate(alg_met):
        ALG = combs[0]
        METRIC = combs[1]
        test_norm, clustering, labels, optic_dict = clean_optics_cluster(X,Y_true,
                                                                MIN_SAMPLES=MIN_SAMPLES,
                                                                METRIC=METRIC,
                                                                ALG=ALG,
                                                                graph='n',
                                                                DF=df)
        dats = (ALG,METRIC,optic_dict,clustering.get_params())
        d.append(dats)
        #plotting
        unique_labels = set(labels)
        core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
        core_samples_mask[clustering.cluster_hierarchy_] = True
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        ROW = COORDS[idx][1]
        COL = COORDS[idx][0]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = labels == k

            xy = test_norm[class_member_mask & core_samples_mask]
            ax[ROW,COL].plot(xy[:, 0],xy[:, 1],"o",markerfacecolor=tuple(col),
                                        markeredgecolor="k",
                                        markersize=14
                                        )

            xy = test_norm[class_member_mask & ~core_samples_mask]
            ax[ROW,COL].plot(xy[:, 0],xy[:, 1],"o",markerfacecolor=tuple(col),
                                        markeredgecolor="k",
                                        markersize=6,
                                            )

            ax[ROW,COL].set_title(f"{METRIC}")
    plt.savefig('Graph3_Comparing_Metrics.png')
    plt.tight_layout()
    plt.show()
    
    cols = ['Clusters',  'Noise',  'Homogeneity',  'Completeness',  'V-measure',  'Adjusted Rand Index',  'Adjusted Mutual Information',  'Silhouette Coefficient']
    tmp_dfs = []
    for info in d: 
        indx = [info[0] + '-'+ info[1]]
        results = info[2]
        cols = info[2].keys()
        xdf = pd.DataFrame.from_records(results, columns = cols, index=indx)
        tmp_dfs.append(xdf)
    dfs = pd.concat(tmp_dfs)
    dfs = dfs.reset_index()
    dfs['Algorithm'] = dfs['index'].apply(lambda x: x.split('-')[0])
    dfs['Metric'] = dfs['index'].apply(lambda x: x.split('-')[1])
    dfs = dfs.drop('index',axis=1)
    dfs.to_csv('Metrics_Cluster_test.csv')

    return dfs




