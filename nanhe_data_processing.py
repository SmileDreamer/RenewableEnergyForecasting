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


def process(path, train_list):
    # 把path下的train_list里的所有文件合并成一个文件
    final_file = pd.DataFrame()
    for i in range(len(train_list)):
        curr_path = path + train_list[i] + '.xls' 
        print('curr_path:', curr_path)
        curr_file = pd.read_excel(curr_path)
        
        #删除包含缺失值的行
        print(curr_file.isnull().any())
        # print(np.isnan(curr_file).any())
        train_null = pd.isnull(curr_file)
        train_null = curr_file[train_null == True]
        print(train_null)
        curr_file.dropna(inplace=True)
        
        final_file = pd.concat([final_file, curr_file], axis = 0)
    
    # 保存final_file
    final_file.rename(columns={'总辐射':'实发辐照度',
                               '直辐射':'辐照度',
                               '环境温度':'温度',
                               '地面气压':'压强',
                               '环境风速':'风速',
                               '环境风向':'风向',
                               '实际发电功率':'实际功率'}, inplace=True)
    final_file.drop(columns={'序号', '场站名', '散辐射', '组件温度', '开机容量'}, axis=1, inplace=True)
    final_file.to_csv(path + 'final_file.csv', index = False)


if __name__== '__main__':
    # Path: Power_prediction_only_code/nanhe_data_processing.py
    # Compare this snippet from Power_prediction_only_code/data_preprocessing.py:
    path = './nanhe_data/'
    year_list = ['2020', '2021', '2022']
    train_list = [['10', '11', '12'], ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'], ['1', '2', '3', '4', '5', '6', '7']]
    
    for i in range(len(year_list)):
        year_path = path + year_list[i] + '/'
        process(year_path, train_list[i])