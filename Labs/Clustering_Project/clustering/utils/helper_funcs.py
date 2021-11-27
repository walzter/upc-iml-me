import sys
from scipy.io.arff import loadarff
import pandas as pd
import tqdm
from tqdm import tqdm
import os

#plotting
import matplotlib.pyplot as plt

# preprocessing: Numerical
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

# preprocessing: Categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#loading a dataset
def load_dataset(FILE_NAME):
    data,meta = loadarff(FILE_NAME)
    df = pd.DataFrame(data)
    return df
#df[x] = df[x].copy().str.decode('utf-8') -> add to cleaning part

#
# Preprocess everythogn
def preprocess_dataset(dataframe,label_col=[],ohe='y'):

    #first we want to separate the target column
    Y_true = (dataframe[label_col].values).reshape(-1,)
    dataframe = dataframe.drop(label_col,axis=1,)

    #now we select the numerical columns
    df_nums = dataframe.select_dtypes(include='number')
    df_cat = dataframe.select_dtypes(include='object')

    # encoding the label
    le = LabelEncoder()
    le = le.fit(Y_true)
    le=le.transform(Y_true)
    Y_true_le = pd.DataFrame(le,columns=['le_class'])

    if ohe=='y':
        #OHE the remaining categorical columns
        ohe = OneHotEncoder()
        cat_data = (df_cat.values).reshape(-1,1)
        ohe = ohe.fit(cat_data)
        #columns
        names = ohe.get_feature_names()
        #ohe_fit.get_feature_names_out() #feature names
        ohe_transf = ohe.transform(cat_data).toarray()
        df_ohe = pd.DataFrame(ohe_transf,columns=names)


    #now we preprocess the numerical one
    data = df_nums.values
    data_test = data.copy()
    X = ('Unscaled data', data_test)
    distributions = [
        ('Unscaled data', X[1]),
        ('standard scaling',
        StandardScaler().fit_transform(X[1])),
        ('min-max scaling',
        MinMaxScaler().fit_transform(X[1])),
        ('max-abs scaling',
        MaxAbsScaler().fit_transform(X[1])),
        ('robust scaling',
        RobustScaler(quantile_range=(25, 75)).fit_transform(X[1])),
        ('power transformation (Yeo-Johnson)',
        PowerTransformer(method='yeo-johnson').fit_transform(X[1])),
        ('power transformation (Box-Cox)',
        PowerTransformer(method='box-cox').fit_transform(X[1])),
        ('quantile transformation (uniform pdf)',
        QuantileTransformer(output_distribution='uniform')
        .fit_transform(X[1])),
        ('quantile transformation (gaussian pdf)',
        QuantileTransformer(output_distribution='normal')
        .fit_transform(X[1])),
        ('sample-wise L2 normalizing',
        Normalizer().fit_transform(X[1])),
    ]

    return distributions, Y_true_le

def get_cat_num_col_stats(PATH='./datasets', graph='no',return_dict_of_cols='n'):
    # all the arff files

    datasets = os.listdir(PATH)
    # dictionary to store values
    cat_num_dict = dict()
    cat_col_dict = dict()
    for file in tqdm(datasets):
        file_name = PATH + "/" + file
        df = load_dataset(file_name)
        tmp = dict(df.dtypes.value_counts())
        #tmp_cols = [(x,idx) for x in enumerate(df.dtypes)]
        cat_num_dict[file] = tmp
        cat_col_dict[file] = [(x,df.dtypes.index[idx]) for idx,x in enumerate(df.dtypes)]

    #converting to dataframe and sorting
    df = pd.DataFrame(cat_num_dict)
    df = df.fillna(0)
    df = df.transpose()
    df = df.reset_index()
    df.columns = ['dataset','object_columns','float64_columns']
    df['pcnt_obj'] = (df['object_columns']/(df['object_columns'] + df['float64_columns'])) * 100
    df['Type'] = ['Categorical' if x >= 51 else 'Numerical' for x in df['pcnt_obj']]
    df['Total_cols'] = df['object_columns'] + df['float64_columns']
    count_numerical = df['float64_columns'].sum()
    count_categorical = df['object_columns'].sum()
    cat_cols = [(x,df.dtypes.index[idx]) for idx,x in enumerate(df.dtypes)]
    total = df['Total_cols'].sum()
    pcnt_numerical = (count_numerical/total)*100
    pcnt_categorical = (count_categorical/total)*100
    labels = ['Numerical','Mostly Numerical','Mostly Categorical','Categorical']
    df['label'] = pd.qcut(df['pcnt_obj'],4,labels=labels)
    print(f"\nThere are {df.shape[0]} datasets in {PATH}. \
            \nTotal Numerical Columns: {count_numerical} -- {pcnt_numerical:.0f}%. \
            \nTotal Categorical Columns: {count_categorical} -- {pcnt_categorical:.0f}%")
    if graph == 'yes':
        df[['Type','Total_cols']].groupby('Type') \
                                                .sum()\
                                                .plot(kind='barh',
                                                      title='Graph 1. Total Number of columns across all datasets',
                                                      xlabel='Number of Datasets');
        plt.show()
        df.groupby('label').count()['dataset'].plot(kind='bar',rot=0,figsize=(20,10),title='Graph 2. Count of types of dataset',
            xlabel='Type',ylabel='Count');
    
    if return_dict_of_cols == 'y':
        return df, cat_col_dict
    else:
        return df
    return df

def get_col_type_nans(PATH):
    cat_nans_cols = []
    for files in os.listdir(PATH):
        # loading the dataset
        df = load_dataset(PATH + '/' + files)
        size_cols = len(df.columns)
        for cols in df.columns:
            col_type = df[cols].dtype
            col_isna = df[cols].isnull().sum()
            pcnt_of_df = ((1/size_cols)*100)
            final_tup = (files,cols,col_type,col_isna,pcnt_of_df)
            cat_nans_cols.append(final_tup)
    columns = ['dataset','col','type','nans','pcnt_of_df']
    df = pd.DataFrame(cat_nans_cols,columns=columns)

    return df

#categorical 
def get_ohe_df(df,cat_col):
       
    #prepping the data:
    cat_data = df[cat_col].values
    cat_data = cat_data.reshape(-1,1)
    #OneHot
    ohe = OneHotEncoder()
    ohe = ohe.fit(cat_data)
    #columns
    names = ohe.get_feature_names()
    #ohe_fit.get_feature_names_out() #feature names
    ohe_transf = ohe.transform(cat_data).toarray()
    #return dataframe with OHE
    df_ohe = pd.DataFrame(ohe_transf,columns=names)
    
    return df_ohe
    
# LabelEncoder
def get_le_df(df,cat_col):
    #prepping the data:
    cat_data = df[cat_col].values
    cat_data = cat_data.reshape(-1,)
    #OneHot
    le = LabelEncoder()
    le = le.fit(cat_data)
    le=le.transform(cat_data)
    df = pd.DataFrame(le,columns=['le_class'])
    return df


# Numerical preprocessing
def num_col_standardize(df, graph='n'):
    ## Data
    #numerical columns only
    df_num = df.copy().drop('class',axis=1)
    #
    data = df_num.values
    data_test = data.copy()
    X = ('Unscaled data', data_test)
    distributions = [
        ('Unscaled data', X[1]),
        ('standard scaling',
        StandardScaler().fit_transform(X[1])),
        ('min-max scaling',
        MinMaxScaler().fit_transform(X[1])),
        ('max-abs scaling',
        MaxAbsScaler().fit_transform(X[1])),
        ('robust scaling',
        RobustScaler(quantile_range=(25, 75)).fit_transform(X[1])),
        ('power transformation (Yeo-Johnson)',
        PowerTransformer(method='yeo-johnson').fit_transform(X[1])),
        ('power transformation (Box-Cox)',
        PowerTransformer(method='box-cox').fit_transform(X[1])),
        ('quantile transformation (uniform pdf)',
        QuantileTransformer(output_distribution='uniform')
        .fit_transform(X[1])),
        ('quantile transformation (gaussian pdf)',
        QuantileTransformer(output_distribution='normal')
        .fit_transform(X[1])),
        ('sample-wise L2 normalizing',
        Normalizer().fit_transform(X[1])),
    ]

    if graph =='y':
        fig, axs = plt.subplots(10,2,figsize=(30,20))
        for idx,x in enumerate(distributions):
            axs[idx,0].hist(X[1])
            axs[idx,0].set_title(X[0])
            
            axs[idx,1].hist(x[1])
            axs[idx,1].set_title(x[0])
        plt.show()
    return distributions


#processing the loaded dataframe
#
# Preprocess everythogn
def preprocess_dataset(dataframe,ohe='n'):
    
    #first we want to separate the target column
    Y_true = (dataframe[label_col].values).reshape(-1,)
    dataframe = dataframe.drop('class',axis=1,)
    
    #now we select the numerical columns
    df_nums = dataframe.select_dtypes(include='number')
    print('nums')
    print(df_nums)
    print(df_nums.shape)
    df_cat = dataframe.select_dtypes(include='object')
    print('cats')
    print(df_cat)
    print(df_cat.shape)
    
    # encoding the label 
    le = LabelEncoder()
    le = le.fit(Y_true)
    le=le.transform(Y_true)
    Y_true_le = pd.DataFrame(le,columns=['le_class'])

    if "class" in df_cat.columns:
        pass
    else: 
        #OHE the remaining categorical columns 
        ohe = OneHotEncoder()
        cat_data = (df_cat.values)#.reshape(-1,1)
        ohe = ohe.fit(cat_data)
        #columns
        names = ohe.get_feature_names()
        #ohe_fit.get_feature_names_out() #feature names
        ohe_transf = ohe.transform(cat_data).toarray()
        df_ohe = pd.DataFrame(ohe_transf,columns=names)
        return df_ohe

    #now we preprocess the numerical one
    data = df_nums.values
    data_test = data.copy()
    X = ('Unscaled data', data_test)
    print(data_test)
    distributions = [
        ('Unscaled data', X[1]),
        ('standard scaling',
         StandardScaler().fit_transform(X[1])),
        ('min-max scaling',
         MinMaxScaler().fit_transform(X[1])),
        ('max-abs scaling',
         MaxAbsScaler().fit_transform(X[1])),
        ('robust scaling',
         RobustScaler(quantile_range=(25, 75)).fit_transform(X[1])),
        ('power transformation (Yeo-Johnson)',
         PowerTransformer(method='yeo-johnson').fit_transform(X[1])),
        ('power transformation (Box-Cox)',
         PowerTransformer(method='box-cox').fit_transform(X[1])),
        ('quantile transformation (uniform pdf)',
         QuantileTransformer(output_distribution='uniform')
         .fit_transform(X[1])),
        ('quantile transformation (gaussian pdf)',
         QuantileTransformer(output_distribution='normal')
         .fit_transform(X[1])),
        ('sample-wise L2 normalizing',
         Normalizer().fit_transform(X[1])),
    ]

    return distributions, df_ohe,Y_true_le
