# -*- coding: utf-8 -*-
import numpy as np
def compute_metrics(true_series, forecast):
    
    ''' first section  the errors based on absolute error'''
    
    
    mae=np.mean(abs(true_series-forecast))
       
    #Compute Relative Absolute Error(RAE), can only range from zero to one
    # true_mean=np.mean(true_series)
    # abs_error_sum=np.sum(np.abs(true_series-forecast))
    # true_deviation_sum=np.sum(np.abs(true_series-true_mean))
    # rae=abs_error_sum/true_deviation_sum
    numerator = np.sum(abs(forecast - true_series))

    denominator = np.sum(abs(np.mean(true_series) - true_series))
    rae=numerator / denominator
    
    #compute Mean Absolute Percetage Error(MAPE)
    # abs_error=(np.abs(true_series-forecast))/true_series
    # mape=np.mean(abs_error)*100
    mape=np.mean(abs((true_series - forecast) / true_series)) * 100
  
    ''' second section the errors based on squered errors '''
    #Compute Mean Squared Error
    # mse = tf.keras.losses.MeanSquaredError()
    # mse=mse(true_series, forecast).numpy()
    mse=np.mean(np.square(true_series-forecast))
    #Compute  Root Mean Squared Error (RMSE)
    #rmse=math.sqrt(mse)
    rmse=np.sqrt(np.mean(np.square(true_series-forecast)))
    #Compute Relative  Squared Error,The output value of RSE is expressed in terms of ratio. It can range from zero to one. 
    true_mean=np.mean(true_series)
    squared_error_sum=np.sum(np.square(true_series-forecast))
    squared_true_dev_sum=np.sum(np.square(true_series-true_mean))
    rse=squared_error_sum/squared_true_dev_sum
    
    #Compute 
    ''' Third section MBE'''
    #Compute mean bias error,This evaluation metric quantifies the overall bias and captures the average bias in the prediction. 
    mbe=np.mean(true_series-forecast)
    
    return mae,rae,mape,mse,rse,mbe,rmse