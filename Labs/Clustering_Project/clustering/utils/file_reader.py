import pandas as pd
import numpy as np
from scipy.io import arff




#
# Comment: This will fail with other dataframes if there 
# isn't a defined "class" column. 
#
#
def load_arff_file(file_name, labels='class'):
    data, meta = arff.loadarff(file_name)
    df = pd.DataFrame(data)
    df_all = df.copy(deep=True)
    df = df.drop(labels=labels, axis=1)

    return df.to_numpy(), df_all.to_numpy(), meta
