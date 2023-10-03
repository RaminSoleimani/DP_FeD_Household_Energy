#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 18:30:08 2023

@author: rs1
"""


import tensorflow as tf
from helper import *
from sklearn.preprocessing import MinMaxScaler
import os
import keras
from tensorflow.keras.utils import plot_model

#load a model
loaded_model=tf.keras.models.load_model("/var/share/rs1/projects/Fed_STLF/BaseModels/model_20230726-161145")



# Specify the path to the CSV file
csv_file_path =  r"/var/share/rs1/LCL_DATA/preparded_houshold_data/block0_MAC000002.csv"

# Get the name of the CSV file
file_name = os.path.basename(csv_file_path)

#read the csv file
df=pd.read_csv(csv_file_path)



#craete series
series=df['energy(kWh/hh)']


# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))

''' getting the train and structural information of the saved models'''

#print model summary
print(loaded_model.summary())

# Get the name of the loss function
loss_function_name = loaded_model.loss
print("Loss function:", loss_function_name)

# Get the name of the optimizer
optimizer_name = loaded_model.optimizer.get_config()['name']
print("Optimizer:", optimizer_name)


# Visualize the model architecture and save it as a file (e.g., model.png)
plot_model(loaded_model, to_file='model.png', show_shapes=True, show_layer_names=True)

import matplotlib.pyplot as plt
import numpy as np




# # Get the training history
# training_history = loaded_model.history

# # Extract the desired metrics
# loss = training_history.history['loss']
# accuracy = training_history.history['accuracy']

# # Create the x-axis values (epochs)
# epochs = np.arange(1, len(loss) + 1)

# # Plot the loss
# plt.plot(epochs, loss, label='Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training Loss')
# plt.legend()
# plt.show()

# # Plot the accuracy
# plt.plot(epochs, accuracy, label='Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Training Accuracy')
# plt.legend()
# plt.show()

#colculate the train size base on split_time variable
split=int(G.SPLIT_TIME*len(series))


#Evaluate and visualize the result of the model
model_performance_visulization(loaded_model,series,scaler,G.WINDOW_SIZE,split)
