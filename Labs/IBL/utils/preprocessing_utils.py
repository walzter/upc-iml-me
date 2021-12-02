from sklearn import preprocessing
import pandas as pd

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
