import pandas as pd

from src.algorithms.ibl1 import IBL1
from src.algorithms.ibl2 import IBL2
from src.algorithms.ibl3 import IBL3
from src.similarities.similarities import euclidean_sim
from src.util.cross_val import cross_val
from util.file_loader import load_dataset, load_datasets

#sys.path.append('./datasetsCBR')

def something():
    print('Hi skal')

def ibl1(filename):
    algorithm = 'ilb1'
    ds_list = load_datasets(filename)
    performance_results, execution_time = cross_val(ds_list, IBL1(euclidean_sim), 'fit', 'predict')
    df = pd.DataFrame(performance_results)
    df.to_csv(f'./logs/{filename}.{algorithm}.performance.csv')
    df = pd.DataFrame(execution_time)
    df.to_csv(f'./logs/{filename}.{algorithm}.times.csv')

def ibl2(filename):
    algorithm = 'ilb2'
    ds_list = load_datasets(filename)
    performance_results, execution_time = cross_val(ds_list, IBL2(euclidean_sim), 'fit', 'predict')
    df = pd.DataFrame(performance_results)
    df.to_csv(f'./logs/{filename}.{algorithm}.performance.csv')
    df = pd.DataFrame(execution_time)
    df.to_csv(f'./logs/{filename}.{algorithm}.times.csv')

def ibl3(filename):
    algorithm = 'ilb3'
    ds_list = load_datasets(filename)
    performance_results, execution_time = cross_val(ds_list, IBL3(euclidean_sim), 'fit', 'predict')
    df = pd.DataFrame(performance_results)
    df.to_csv(f'./logs/{filename}.{algorithm}.performance.csv')
    df = pd.DataFrame(execution_time)
    df.to_csv(f'./logs/{filename}.{algorithm}.times.csv')

def iblflow():
    # used soybean, credit-a, satimage
    filename = 'soybean'
    ibl1(filename)
    ibl2(filename)
    ibl3(filename)

if __name__ == '__main__':
    iblflow()
