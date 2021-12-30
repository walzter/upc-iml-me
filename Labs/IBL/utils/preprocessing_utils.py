from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.svm import LinearSVC
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

def normalise_df(data_df_wo_class, meta):
  categorical_col_names = [meta.names()[:-1][i] for i in [index for index, type in enumerate(meta.types()[:-1]) if type == 'nominal']]
  numerical_data_df_only = data_df_wo_class.drop(categorical_col_names, axis=1)
  data_only = numerical_data_df_only.values # returns a numpy array
  min_max_scaler = preprocessing.MinMaxScaler()
  scaled_data_only = min_max_scaler.fit_transform(data_only)
  scaled_data_df = pd.DataFrame(scaled_data_only, columns=list(numerical_data_df_only))
  return scaled_data_df

def preprocess_data(orig_data_df, meta):
  numerical_col_names = [meta.names()[:-1][i] for i in [index for index, type in enumerate(meta.types()[:-1]) if type == 'numeric']]
  if (len(numerical_col_names) != 0):
    scaled_data_df = normalise_df(orig_data_df.drop(list(orig_data_df)[-1], axis=1), meta)
    # Appending categorical columns and classification column again
    for col in list(orig_data_df):
      if col not in list(orig_data_df.select_dtypes(include='number')):
        scaled_data_df[col] = orig_data_df[col]
    # At this moment, all numerical data is normalised to [0, 1] and all categorical remains as is
    # OHE not required as we're calculating the distance separately, for categorical data
    return scaled_data_df
  else:
    return orig_data_df

def select_features(orig_data_df, selection_method, threshold_or_no_of_features):
  oe = OrdinalEncoder()
  oe.fit(orig_data_df)
  transformed_data_df = oe.transform(orig_data_df)

  imp = SimpleImputer(missing_values=np.nan, strategy='mean')
  imp = imp.fit(transformed_data_df)
  transformed_data_df = imp.transform(transformed_data_df)

  X = transformed_data_df[:, :len(transformed_data_df[0]) - 1]
  Y = transformed_data_df[:, len(transformed_data_df[0]) - 1]
  if (selection_method == 'variance_selection'):
    selector = VarianceThreshold(threshold = threshold_or_no_of_features)
    sel = selector.fit(X)
    sel_index = sel.get_support()
    selected_features = [orig_data_df.columns[i] for i in (sel_index == True).nonzero()][0].values
    return selected_features
  elif (selection_method == 'rfe'):
    svc = LinearSVC(max_iter = 10000)
    svc.fit(X, Y)
    selector = RFE(svc, n_features_to_select = threshold_or_no_of_features, step = 1)
    selector = selector.fit(X, Y)
    selected_features = [orig_data_df.columns[i] for i in (selector.support_ == True).nonzero()][0].values
    return selected_features
  else:
    print('Unsupported feature selection method chosen [' + str(selection_method) + '], skipping feature selection step')
  return None

def prune_non_essential_features(orig_data_df, orig_test_data_df, selected_features):
  pruned_features_df = pd.DataFrame()
  pruned_features_test_df = pd.DataFrame()
  for col in selected_features:
    pruned_features_df[col] = orig_data_df[col].values
    pruned_features_test_df[col] = orig_test_data_df[col].values
  pruned_features_df[orig_data_df.columns[-1]] = orig_data_df[orig_data_df.columns[-1]].values
  pruned_features_test_df[orig_test_data_df.columns[-1]] = orig_test_data_df[orig_test_data_df.columns[-1]].values
  return pruned_features_df, pruned_features_test_df