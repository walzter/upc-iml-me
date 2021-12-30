import glob
import os

import pandas as pd
from scipy.io.arff import loadarff
from sklearn import preprocessing

ROOT_FOLDER = f'{os.getcwd()}/..'


def read_arff(file_name):
    data, meta = loadarff(file_name)
    df = pd.DataFrame(data)
    return df


def load_dataset(ds_name="", index=0, path="datasetsCBR"):
    ds_folder = f'{ROOT_FOLDER}/{path}/{ds_name}/*{index}'
    ds_train = read_arff(glob.glob(f'{ds_folder}.train.arff')[0])
    ds_train = ds_train.dropna()
    ds_test = read_arff(glob.glob(f'{ds_folder}.test.arff')[0])
    ds_test = ds_test.dropna()

    label_col = ds_train.columns[-1:]
    le = preprocessing.LabelEncoder()
    train_encoded_label = le.fit_transform(ds_train[label_col])
    test_encoded_label = le.fit_transform(ds_test[label_col])

    return {
        'x_train':  ds_train.drop(label_col, axis=1),
        'y_train':  train_encoded_label,
        'x_pred':  ds_test.drop(label_col, axis=1),
        'y_pred':  test_encoded_label
    }


def load_datasets(ds_name="", path="datasetsCBR"):
    ds = []
    for i in range(0, 10):
        ds.append(load_dataset(ds_name, index=i, path=path))

    return ds
