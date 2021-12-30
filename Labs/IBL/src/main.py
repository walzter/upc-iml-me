import pandas as pd

from src.algorithms.ibl3 import IBL3 as IBL1
from src.similarities.similarities import euclidean_sim
from src.util.cross_val import cross_val
from util.file_loader import load_dataset, load_datasets

#sys.path.append('./datasetsCBR')

def something():
    print('Hi skal')

def iblflow():
    filename = 'soybean'
    algorithm = 'ilb3'
    ds_list = load_datasets(filename)
    performance_results, execution_time = cross_val(ds_list, IBL1(euclidean_sim), 'fit', 'predict')
    df = pd.DataFrame(performance_results)
    df.to_csv(f'./logs/{filename}.{algorithm}.performance.csv')
    df = pd.DataFrame(execution_time)
    df.to_csv(f'./logs/{filename}.{algorithm}.times.csv')

if __name__ == '__main__':
    iblflow()
