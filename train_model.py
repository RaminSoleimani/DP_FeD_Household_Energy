# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
from helper import *                   
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
import time

# Specify the path to the CSV file
csv_file_path = r"/var/share/rs1/LCL_DATA/preparded_houshold_data/block0_MAC000002.csv"

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))

# Get the name of the CSV file
file_name = os.path.basename(csv_file_path)

#read the csv file
df=pd.read_csv(csv_file_path)



#craete series
series=df['energy(kWh/hh)']

#get the window size
window_size=G.WINDOW_SIZE




# Reshape the data to a 2D array as required by the scaler
series = series.values.reshape(-1, 1)

norm_series = scaler.fit_transform(series)





dataset_train,data_val,dataset_valid_y,split=dataset_train_preparation(csv_file_path,scaler,window_size)

# Test your uncompiled modeltaset! :)")
    
#history=adjust_learning_rate(dataset_train)
#uncompiled_model = create_uncompiled_model()

# try:
#     uncompiled_model.predict(dataset_train)
# except:
#     print("Your current architecture is incompatible with the windowed dataset, try adjusting it.")
# else:
#     print("Your current architecture is compatible with the windowed da



# Plot the loss for every LR
# plt.semilogx(history.history["lr"], history.history["loss"])
# plt.axis([1e-6, 1, 0, 30])

model=create_model(learning_rate=9e-2,net_arch='CONVBILSTM',window_size=window_size)

# get the start time
st = time.time()
# Train it
history = model.fit(dataset_train, epochs=100,callbacks=[tf.keras.callbacks.History()])
# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

model_save(model)


# evaluate and visualize the performance of the model
model_performance_visulization(model,series,scaler,window_size,split)
