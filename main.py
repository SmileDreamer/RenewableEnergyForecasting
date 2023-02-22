from Load_Save_Data import Load_Data, Save_Data
from Data_Process import Data_Process
from Train_Predict import Train_Predict

import numpy as np
from sklearn import preprocessing # 数据归一化
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime

if __name__ == "__main__":
    #1. 加载数据
    print( '加载数据=======', datetime.now() )
    data_train, data_test = Load_Data() # 9, 8

    power_mean = np.mean( data_train['实际功率'] )
    power_std = np.std( data_train['实际功率'] )
    
    #2. 特征工程
    print( '处理数据=======', datetime.now() )
    data_train, data_test = Data_Process( data_train, data_test )

    #测试代码 begin
    #data_train = data_train.iloc[ :10000, : ] #test
    #data_test = data_test.iloc[ :10000, : ] #test
    #测试代码 end
    
    # 保存ground truth:
    gt_irradiance = data_test['实发辐照度']
    gt_power = data_test['实际功率']


    #3. 训练模型, 预测功率
    print( '启动预测=======', datetime.now() )
    power_predict = Train_Predict( data_train, data_test )
    #将归一化数据恢复为正常值
    power_predict = power_predict * power_std + power_mean
    print( '预测完成=======', datetime.now() )
    
    #4. 输出gt_power和power_predict之间的损失：MAE, MSE, RMSE
    gt_irradiance = gt_irradiance.values
    gt_power = gt_power.values
    print( 'MAE: ', mean_absolute_error( gt_power, power_predict ) )
    print( 'MSE: ', mean_squared_error( gt_power, power_predict ) )
    print( 'RMSE: ', np.sqrt( mean_squared_error( gt_power, power_predict ) ) )
    print( 'R2: ', r2_score( gt_power, power_predict ) )
    
    #5. 保存结果数据
    print( '保存预测结果=======', datetime.now() )
    Save_Data( power_predict )
    
    
    
    
    
    
    
    
    
    
    
    
    
    # 用matplotlib画gt_power的柱状图，并保存成jpg:
    plt.figure( figsize = (20, 10) )
    plt.bar( range( len(gt_power) ), gt_power, width = 0.5, color = 'b', label = 'gt_power' )
    plt.bar( range( len(power_predict) ), power_predict, width = 0.5, color = 'r', label = 'power_predict' )
    plt.legend()
    plt.savefig( 'gt_power.pdf' )
    plt.close()
    
    print( '保存完成，退出程序=======', datetime.now() )
