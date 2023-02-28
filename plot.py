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
    gt_power = pd.read_csv('./nanhe_data/test_1.csv')
    gt_power = gt_power['实际功率']
    predict_power = pd.read_csv('./nanhe_data/result_total_ensemble_21.csv')
    
    # 绘制gt_power和predict_power的散点图
    plt.figure(figsize=(10, 10))
    plt.scatter(gt_power, predict_power, s=1)
    plt.xlabel('gt_power')
    plt.ylabel('predict_power')
    plt.savefig('./nanhe_data/compare.pdf')
    plt.close()