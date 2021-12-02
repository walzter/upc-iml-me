import os 
import pandas as pd
import numpy as np
from scipy.io.arff import loadarff

PATH = './data/'
FOLD_AFFIX = '.fold.00000'
TEST_SUFFIX = '.test.arff'
TRAIN_SUFFIX = '.train.arff'

def get_working_files(NAME):
    '''
    Input --> Name of the folder to use 
    Output --> list of training and testing file names 
    '''
    training_list = []
    testing_list = []
    for file in os.listdir(f"./data/{NAME}"):
        if 'train' in file: 
            training_list.append(file)
        else: 
            testing_list.append(file)
    assert len(testing_list) == len(training_list)
    print('Training == Testing')
    return sorted(training_list), sorted(testing_list)

def load_dataset(FILE_NAME):
  data, meta = loadarff(FILE_NAME)
  df = pd.DataFrame(data)
  cleaner = lambda x: x.decode('utf-8') if isinstance(x, str) else x
  vcleaner = np.vectorize(cleaner)
  to_clean = lambda x: vcleaner(x)
  vto_clean = np.vectorize(to_clean)
  cleaned_df = df.apply(vto_clean)
  return cleaned_df, meta

def get_test_and_train_data_for_fold(dataset_name, fold):
  train_file_name = PATH + dataset_name + '/' + dataset_name + FOLD_AFFIX + str(fold) + TRAIN_SUFFIX
  x_train, meta = load_dataset(train_file_name)
  #x_train = x_train.replace(np.nan, None)

  test_file_name = PATH + dataset_name + '/' + dataset_name + FOLD_AFFIX + str(fold) + TEST_SUFFIX
  x_test, meta = load_dataset(test_file_name)
  #x_test = x_test.replace(np.nan, None)

  return x_train, x_test, meta