import glob
import os
import sys

from src.algorithms.ibl1 import IBL1
from src.similarities.similarities import euclidean_sim
from src.util.cross_val import cross_val
from util.file_loader import load_dataset, load_datasets


#sys.path.append('./datasetsCBR')

def something():
    print('Hi skal')


if __name__ == '__main__':
    ds_list = load_datasets('satimage')
    cross_val(ds_list, IBL1(euclidean_sim), 'fit', 'predict')
