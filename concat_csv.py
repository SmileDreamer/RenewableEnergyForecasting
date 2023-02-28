import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold
import csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import sys
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import time

if __name__== '__main__':
    # Path: Power_prediction_only_code/nanhe_data_processing.py
    # Compare this snippet from Power_prediction_only_code/data_preprocessing.py:
    path = './nanhe_data/'
    file_list = ['train_1.csv', 'train_2.csv']
    
    final_file = pd.DataFrame()
    file1 = pd.read_csv(path + file_list[0])
    file2 = pd.read_csv(path + file_list[1])
    final_file = pd.concat([file1, file2], axis = 0)
    final_file.to_csv(path + 'final_file.csv', index = False)
        