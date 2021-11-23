# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys 
sys.path.append('./algorithms')
sys.path.append('./utils')
from utils.helper_funcs import *
from utils.helper_funcs import load_dataset
from utils.helper_funcs import get_cat_num_col_stats
from utils.helper_funcs import get_ohe_df, get_le_df
from utils.helper_funcs import num_col_standardize


from scipy.io import arff
from sklearn.cluster import OPTICS
import numpy as np
import pandas as pd

from algorithms.optics import *
from algorithms.fuzzycmeans import FuzzyCmeans
from algorithms import kmeans, fuzzycmeans, kmodes, optics
from algorithms.kmeans import Kmeans
from algorithms.kmodes import Kmodes
from utils.file_reader import load_arff_file

# setting precision
pd.options.display.float_format = '{:,.2f}'.format

'''
# single file first
PATH_TO_DATA = './datasets'
FILE_NAME = 'iris.arff'
FULL_PATH = PATH_TO_DATA + '/' + FILE_NAME
''' 

CREDIT_A_PATH = "./datasets/credit-a.arff"
HEPATITIS_PATH = "./datasets/hepatitis.arff"
VOWEL_PATH = "./datasets/vowel.arff"
HEART_C_PATH = "./datasets/heart-c.arff"
ZOO_PATH = "./datasets/zoo.arff"
WINE_PATH = "./datasets/wine.arff"
ADULTS_PATH = "./datasets/adult.arff"

#data_array, data_array_with_classes, meta = load_arff_file(WINE_PATH)

'''
df = load_dataset(FULL_PATH)

# sepparating the dataframe into numerical and categorical
X, Y_true_le = preprocess_dataset(df, label_col=['class'],ohe='n')

#scale numerical cols - 9 different ones
num_standardized = num_col_standardize(df,graph='n')

#OPTICS - clustering #only with one of the valuess from above
test_norm, cluster, labels, optic_dict = clean_optics_cluster(num_standardized,
                                                              MIN_SAMPLES=25,
                                                              METRIC='minkowski',
                                                              ALG='auto',
                                                              DF=df,
                                                              graph='n')
'''
#clustering = optics.perform_clustering(data_array)

#kmeans.perform_clustering(data_array)

#kmeans1 = Kmeans()
#kmeans1.fit(data_array)
#data_sample = np.reshape(data_array[0], (1, len(data_array[0])))
#print(kmeans1.predict(data_sample))


#kmodes = Kmodes()
#kmodes.fit(data_array, meta)
#kmodes.print_stats(data_array_with_classes)

#fcmeans = FuzzyCmeans()
#u_matrix, centroids = fcmeans.fit(data_array[:5])
#print(u_matrix)
#print(centroids)

#kmeans.perform_kmeans()

def control_commands():
    user_input = -1
    while user_input != '0':
        user_input = input('Please select type the number related to the action you want to perform.\n'
                           + ' 1 - Run Optics algorithm\n 2 - Run K-Means Algorithm\n 3 - Run K-Modes algorithm\n'
                           + ' 4 - Run Fuzzy C-Means algorithm\n0 - Exit\n ->')

        if user_input == '1':
            optics.perform_optics()

        if user_input == '2':
            kmeans.perform_kmeans()

        if user_input == '3':
            kmodes.perform_kmodes()

        if user_input == '4':
            fuzzycmeans.perform_fuzzycmeans()

        print(user_input)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    control_commands()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
