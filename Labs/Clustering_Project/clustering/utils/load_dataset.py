from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import pandas as pd
from utils.helper_funcs import load_dataset

#preprocessing function: Iris 
def load_iris(path='./datasets/iris.arff'):
    df = load_dataset(path)
    df['class'] = df['class'].apply(lambda x: x.decode('utf-8'))
    Y_true = df['class'].values
    
    #encoding the label 
    le = LabelEncoder()
    Y_true = le.fit_transform(Y_true)
    Y_true = pd.DataFrame(Y_true, columns=['le_class'])
    
    #normalize the values 
    df_num = df.drop('class',axis=1)
    normalized_vals = normalize(df_num.values)
    #only standard scaling
    scale = StandardScaler()
    scaled_vals = scale.fit_transform(normalized_vals)
    
    return df, scaled_vals, Y_true
    

def load_kropt(path='./datasets/kropt.arff'):
    #loading
    df = load_dataset(path)
    # the different columns here
    numeric_cols = ['white_king_row','white_rook_row','black_king_row']
    cat_cols = ['white_king_col','white_rook_col','black_king_col']
    # subsetting the dataframe 
    numeric_data = df[numeric_cols]
    cat_data = df[cat_cols]
    #utf8 decoding all the values 
    df = df.applymap(lambda x: x.decode('utf-8'))
    # sepparating the label 
    Y_true = df['game'].values
    #label encoder
    le = LabelEncoder()
    Y_true = le.fit_transform(Y_true)
    Y_true = pd.DataFrame(Y_true, columns=['le_class'])
    #ohe
    ohe = OneHotEncoder()
    ohe = ohe.fit(cat_data)
    names = ohe.get_feature_names()
    ohe_transf = ohe.transform(cat_data).toarray()
    df_ohe = pd.DataFrame(ohe_transf,columns=names)
    #numericalvalues 
    normalized_vals = normalize(numeric_data)
    #standardscaler
    scale = StandardScaler()
    scaled_vals = scale.fit_transform(normalized_vals)
    scaled_vals = pd.DataFrame(scaled_vals, columns=numeric_cols)
    transformed_df = scaled_vals.join([df_ohe,Y_true])
    return transformed_df, scaled_vals, Y_true

def load_soybean(path='./datasets/soybean.arff'):
    df = load_dataset(path)
    #decode the values from bytes to str
    df = df.applymap(lambda x: x.decode('utf-8'))
    #replace the question marks with np.nans
    df = df.replace("?",np.nan)
    #fill with 0's
    df.fillna(0)
    # we only lose 121 rows of data, not too bad, delete it all 
    df = df.dropna(axis=0)
    Y_true = df['class']
    #label encoding 
    le = LabelEncoder()
    Y_true = le.fit_transform(Y_true)
    Y_true = pd.DataFrame(Y_true, columns=['le_class'])
    #OneHotEncoding
    cat_data = df.drop('class',axis=1)
    ohe = OneHotEncoder()
    ohe = ohe.fit(cat_data)
    names = ohe.get_feature_names()
    ohe_transf = ohe.transform(cat_data).toarray()
    df_ohe = pd.DataFrame(ohe_transf,columns=names)
    transformed_df = df_ohe.join([Y_true])
    return transformed_df, Y_true