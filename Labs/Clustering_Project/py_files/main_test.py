import sys 
sys.path.append('./py_files')
from helper_funcs import *
from optics import *
from helper_funcs import load_dataset
from helper_funcs import get_cat_num_col_stats
from helper_funcs import get_ohe_df, get_le_df
from helper_funcs import num_col_standardize
#
import pandas as pd
import os
# setting precision
pd.options.display.float_format = '{:,.2f}'.format

# single file first
PATH_TO_DATA = './datasets'
FILE_NAME = 'iris.arff'
FULL_PATH = PATH_TO_DATA + '/' + FILE_NAME

# qcut of all the dataframes dtypes count - Categorical Vs. Numerical
#cat_num_df = get_cat_num_col_stats(PATH=PATH_TO_DATA,graph='no')
col_type_nans = get_col_type_nans(PATH_TO_DATA)
# Loading dataset 
df = load_dataset(FULL_PATH)

# sepparating the dataframe into numerical and categorical
df_num,df_cat = df.copy().drop('class',axis=1), df['class'].copy()

#transform categorical cols (ohe -> vals to cols of 0/1, le->classes(0,1,2..))
df_cat_ohe = get_ohe_df(df, 'class')
df_cat_le = get_le_df(df, 'class')

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