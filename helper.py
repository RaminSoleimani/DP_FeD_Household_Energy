# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Mon May 29 16:23:53 2023

@author: 35385
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os
import pandas as pd
import datetime
import numpy as np
import math

#golbal variable
class G:
    SPLIT_TIME = 0.9  #25600
    WINDOW_SIZE = 8
    BATCH_SIZE = 15 #20
    SHUFFLE_BUFFER_SIZE = 1000
    
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

def dataset_train_preparation(csv_file_path,scaler,window_size):
    #read the csv file
    df=pd.read_csv(csv_file_path)
    #craete series
    series=df['energy(kWh/hh)']
    
    # Reshape the data to a 2D array as required by the scaler
    series = series.values.reshape(-1, 1)
    
    #fit scaler
    norm_series = scaler.fit_transform(series)
    
    #colculate the train size base on split_time variable
    split=int(G.SPLIT_TIME*len(series))
    
    #split train and validation set
    time_train, series_train, time_valid, series_valid=train_val_split(range(0,len(series)), 
                                                                       norm_series, time_step=split)
    
    # #window the train dataset
    dataset_train=windowed_dataset(series_train, window_size=window_size,batch_size=G.BATCH_SIZE, shuffle_buffer=G.SHUFFLE_BUFFER_SIZE)
    
   
    dataset_valid_y=[]
    
    for i in range(split,len(series)-window_size):
                dataset_valid_y.append(norm_series[i+window_size]) 
   
    ds = tf.data.Dataset.from_tensor_slices(series_valid)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    data_val = ds.batch(G.BATCH_SIZE).prefetch(1)
    
    return dataset_train,data_val,dataset_valid_y,split

# # Specify the path to the CSV file
# csv_file_path = r"C:\Repositories\LCLDataPreparation\household_data\block1_MAC001628.csv"

# # Get the name of the CSV file
# file_name = os.path.basename(csv_file_path)

# #read the csv file
# df=pd.read_csv(csv_file_path)

# dataset=windowed_dataset(df['energy(kWh/hh)'], window_size=8, batch_size=12, shuffle_buffer=1000)

#create uncompiled model
def create_uncompiled_model_LSTM():
    
    model = tf.keras.models.Sequential([ 
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                      input_shape=[None]),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
        tf.keras.layers.Dense(32),
        tf.keras.layers.Dense(1)
    ]) 
    
    return model
def create_uncompiled_model_CONVLSTM(window_size):
      
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                      strides=1,
                      activation="relu",
                      padding='causal',
                      input_shape=[window_size, 1]),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1)])
    
    return model
def create_uncompiled_model_CONVBILSTM(window_size):
      
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                      strides=1,
                      activation="relu",
                      padding='causal',
                      input_shape=[window_size, 1]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256)),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1)])
    
    return model

def create_uncompiled_model_CONVLSTM(window_size):
      
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                      strides=1,
                      activation="relu",
                      padding='causal',
                      input_shape=[window_size, 1]),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1)])
    
    return model

#Processing the data
def train_val_split(time, series, time_step):

    time_train = time[:time_step]
    series_train = series[:time_step]
    time_valid = time[time_step:]
    series_valid = series[time_step:]

    return time_train, series_train, time_valid, series_valid


#function to save the trained model
def model_save(model):
    # Generate a unique identifier using the current date and time
     timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
     
     # Define the directory where you want to save the model
     save_path = r'/var/share/rs1/projects/Fed_STLF/BaseModels/model_{}'.format(timestamp)
     
     # Save the model with the unique identifier
     model.save(save_path)

def adjust_learning_rate(dataset,net_arch,window_size):
    
    if net_arch=='LSTM':
        model = create_uncompiled_model_LSTM()
    elif net_arch=='CONVLSTM':
        model = create_uncompiled_model_CONVLSTM(window_size)
    elif net_arch=='CONVBILSTM':
        model=create_uncompiled_model_CONVBILSTM(window_size)
    
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-6 * 10**(epoch / 20))
    
    ### START CODE HERE
    
    # Select your optimizer
    optimizer = tf.keras.optimizers.SGD(momentum=0.9)
    
    
    # Compile the model passing in the appropriate loss
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer, 
                  metrics=["mae"]) 
    
    ### END CODE HERE
    
    history = model.fit(dataset, epochs=100, callbacks=[lr_schedule])
    model_save(model)
   
    return history



    

 #create a model   
def create_model(learning_rate,net_arch,window_size):

    tf.random.set_seed(51)
    
    if net_arch=='LSTM':
        model = create_uncompiled_model_LSTM()
    elif net_arch=='CONVLSTM':
        model = create_uncompiled_model_CONVLSTM(window_size)
    elif net_arch=='CONVBILSTM':
        model=create_uncompiled_model_CONVBILSTM(window_size)

    ### START CODE HERE

    model.compile(loss='mae', optimizer=tf.keras.optimizers.SGD(momentum=0.9,learning_rate = 4e-4),
                  metrics=['mae','mse'])  
    
    # model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(learning_rate=4e-4),
    #                metrics=['mae','mse'])
    tf.keras.optimizers.Adam(learning_rate=4e-4)
    ### END CODE HERE

    return model    
    
    
    
#Evaluating and forcast

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
#test 
# true=np.array([1,2,3,4])
# predicted=np.array([0.6,1.3,2.2,3.1]) #.4 .7 .8 .9=2.8  1.5 .5 .5 1.5=4
# print(compute_metrics(true,predicted))

#Forecast by the model

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast
    
    

#The next cell includes a bunch of helper functions to generate and plot the time series:
    
def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(False)   

#Visualize the model performance
def model_performance_visulization(model,series,scaler,window_size,split_time):
    
    # Reshape the data to a 2D array as required by the scaler
    series = series.values.reshape(-1, 1)
    norm_series = scaler.fit_transform(series)
    # Compute the forecast for all the series
    rnn_forecast = model_forecast(model, norm_series, window_size)
    # Reverse the standardization
    predictions = scaler.inverse_transform(rnn_forecast).squeeze()
    #test
    series=series.squeeze()
    print('series size:',len(series),series.shape)
    print('predictions size:',len(predictions),predictions.shape)
    print('valuessss',predictions[20],series[20])
    
    
    mae,rae,mape,mse,rse,rmse,mbe=compute_metrics(series[split_time:], predictions[split_time -window_size:-1])
    print('meeeeeeeee',mae,rae,mape,mse,rse,mbe)
    #Plot actual and predicted values for the validation set
    plt.figure(figsize=(10, 6))
    
    plt.plot(series[split_time:], label='Actual')
    plt.plot(predictions[split_time -window_size:-1], label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Load')
    
    # # Add text boxes with MSE and MAE values
    plt.text(0.05, 0.95, f"mae: {mae:.2f}", transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.text(0.05, 0.9, f"rae: {rae:.2f}", transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.text(0.05, 0.85, f"mape: {mape:.2f}", transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.text(0.15, 0.95, f"mse: {mse:.2f}", transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.text(0.15, 0.9, f"rse: {rse:.2f}", transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.text(0.15, 0.85, f"rmse: {rmse:.2f}", transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.text(0.25, 0.95, f"mbe: {mbe:.2f}", transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(series[split_time:split_time+200], label='Actual')
    plt.plot(predictions[split_time -window_size:split_time -window_size+200], label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Load')
    
    # # Add text boxes with MSE and MAE values
    plt.text(0.05, 0.95, f"mse: {mse:.2f}", transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    
    plt.text(0.05, 0.85, f"mae: {mae:.2f}", transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.legend()
    plt.show()
    
    '''Error analysis'''
    
    
    
    
    
    
    
    
    