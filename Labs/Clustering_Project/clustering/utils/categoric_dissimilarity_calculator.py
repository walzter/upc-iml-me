# Util method to calculate dissimilarity between two data points by frequency of occurrence of the feature value
def get_dissim_categoric(data_row, cluster_center, feature_size, type_metadata):
    dissim = 0
    if len(data_row) < len(cluster_center):
        print('ERROR - cannot calculate dissimilarity as length doesn\'t match with cluster center')
    else:
        for feature_no in range(0, feature_size):
            if (type_metadata.types()[feature_no] == 'nominal') and (
                    data_row[feature_no] != cluster_center[feature_no]):
                dissim += 1
    return dissim
