# -*- coding: utf-8 -*-

import collections
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd

import tensorflow_federated as tff

print("helloooooooooooooooooooooo")

@tff.federated_computation
def hello_world():
  return 'Hello, World!'

hello_world()


#load data for  multi tier  federated learning

with open("list_of_clusters", "rb") as fp:   # Unpickling
    clustered_multi_federated_data = pickle.load(fp) 
#golbal variable
class G:
    SPLIT_TIME = 0.9  #25600
    WINDOW_SIZE = 8
    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 1000

#Creat dataset for a single user
def create_dataset(series, window_size):
    X = []
    y = []
    for i in range(len(series) - window_size):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size])
    X=np.array(X)
    y=np.array(y)


    split_index = int(len(X) * G.SPLIT_TIME)

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    # Convert the data into TensorFlow Datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    # Shuffle and batch the datasets
    batch_size = G.BATCH_SIZE
    train_dataset = train_dataset.shuffle(len(X_train)).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, test_dataset

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

def train_test_multi_fed_data(clustered_multi_federated_data,scaler):
    window_size=G.WINDOW_SIZE
    multi_tier_fed_train_data=[]
    multi_tier_fed_test_data=[]
    for item in clustered_multi_federated_data:
        train_data_lis=[]
        test_data_lis=[]
        for series in item:
            series=series.values.reshape(-1,1)
            norm_series=scaler.fit_transform(series)
            train_dataset,test_dataset=create_dataset(norm_series,window_size)
            train_data_lis.append(train_dataset)
            test_data_lis.append(test_dataset)
            
        multi_tier_fed_train_data.append(train_data_lis)
        multi_tier_fed_test_data.append(test_data_lis)
    return multi_tier_fed_train_data, multi_tier_fed_test_data
multi_tier_fed_train_dataset, multi_tier_fed_test_dataset=train_test_multi_fed_data(clustered_multi_federated_data,scaler)
print(len(multi_tier_fed_train_dataset))
print(multi_tier_fed_train_dataset[1][1])


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
        tf.keras.layers.Dense(1)
    ])
    return model

def model_fn():
  # We _must_ create a new model here, and _not_ capture it from an external
  keras_model = create_uncompiled_model_CONVLSTM(G.WINDOW_SIZE)
  return tff.learning.models.from_keras_model(
      keras_model,
      input_spec=multi_tier_fed_train_dataset[0][0].element_spec,
      loss=tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.MeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()])

@tff.tf_computation
def server_init():
  model = model_fn()
  return model.trainable_variables

@tff.federated_computation
def initialize_fn():
  return tff.federated_value(server_init(), tff.SERVER)
str(initialize_fn.type_signature)

print( type(initialize_fn()))
@tf.function
def client_update(model, dataset, server_weights, client_optimizer):
  """Performs training (using the server model weights) on the client's dataset."""
  # Initialize the client model with the current server weights.(I think this comment is not correct as below code is capturing the clients weights from the last round)
  client_weights = model.trainable_variables
  # Assign the server weights to the client model.
  tf.nest.map_structure(lambda x, y: x.assign(y),
                        client_weights, server_weights)

  # Use the client_optimizer to update the local model.
  for batch in dataset:
    with tf.GradientTape() as tape:
      # Compute a forward pass on the batch of data
      outputs = model.forward_pass(batch)

    # Compute the corresponding gradient
    grads = tape.gradient(outputs.loss, client_weights)
    grads_and_vars = zip(grads, client_weights)

    # Apply the gradient using a client optimizer.
    client_optimizer.apply_gradients(grads_and_vars)

  return client_weights
@tf.function
def server_update(model, mean_client_weights):
  """Updates the server model weights as the average of the client model weights."""
  model_weights = model.trainable_variables
  # Assign the mean client weights to the server model.
  tf.nest.map_structure(lambda x, y: x.assign(y),
                        model_weights, mean_client_weights)
  return model_weights

#creating tff type for dataset and model weights

whimsy_model = model_fn()

tf_dataset_type = tff.SequenceType(whimsy_model.input_spec)

model_weights_type = server_init.type_signature.result

str(model_weights_type)

@tff.tf_computation(tf_dataset_type, model_weights_type)
def client_update_fn(tf_dataset, server_weights):
  model = model_fn()
  client_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
  return client_update(model, tf_dataset, server_weights, client_optimizer)


@tff.tf_computation(model_weights_type)
def server_update_fn(mean_client_weights):
  model = model_fn()
  return server_update(model, mean_client_weights)

federated_server_type = tff.FederatedType(model_weights_type, tff.SERVER)
federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)

@tff.federated_computation(federated_server_type, federated_dataset_type)
def next_fn(server_weights, federated_dataset):
  # Broadcast the server weights to the clients.
  server_weights_at_client = tff.federated_broadcast(server_weights)

  # Each client computes their updated weights.
  client_weights = tff.federated_map(
      client_update_fn, (federated_dataset, server_weights_at_client))

  # The server averages these updates.
  mean_client_weights = tff.federated_mean(client_weights)

  # The server updates its model.
  server_weights = tff.federated_map(server_update_fn, mean_client_weights)

  return server_weights

@tff.tf_computation
def global_server_init():
  model = model_fn()
  return model.trainable_variables

@tff.federated_computation
def global_initialize_fn():
  return tff.federated_value(server_init(), tff.SERVER)
global_server_state=global_initialize_fn()

print(global_server_state[0])

#create tff iterative templates for the number of the federations

number_of_federation=len(multi_tier_fed_test_dataset)

print(number_of_federation)

list_of_tff_tmpl=[]

for i in range(0,number_of_federation):
    list_of_tff_tmpl.append(tff.templates.IterativeProcess(
    initialize_fn=initialize_fn,
    next_fn=next_fn))

print(type(list_of_tff_tmpl[0]))
print(len(list_of_tff_tmpl))

@tf.function
def global_client_update(model, dataset):
  """Performs training (using the server model weights) on the client's dataset."""
  # Initialize the client model with the current server weights.(I think this comment is not correct as below code is capturing the clients weights from the last round)
  global_client_weights = model.trainable_variables
  # Assign the server weights to the client model.
  tf.nest.map_structure(lambda x, y: x.assign(y),
                        global_client_weights, dataset)
  return global_client_weights

model_weights_type = server_init.type_signature.result
@tff.tf_computation(model_weights_type)
def global_client_update_fn(tf_dataset):
  model = model_fn()
  return global_client_update(model, tf_dataset)

'''Global server update fuctions'''

@tf.function
def global_server_update(model, mean_client_weights):
  """Updates the server model weights as the average of the client model weights."""
  model_weights = model.trainable_variables
  # Assign the mean client weights to the server model.
  tf.nest.map_structure(lambda x, y: x.assign(y),
                        model_weights, mean_client_weights)
  return model_weights

@tff.tf_computation(model_weights_type)
def global_server_update_fn(mean_client_weights):
  model = model_fn()
  return global_server_update(model, mean_client_weights)


federated_server_type = tff.FederatedType(model_weights_type, tff.SERVER)
federated_dataset_type = tff.FederatedType(model_weights_type, tff.CLIENTS)


@tff.federated_computation(federated_server_type, federated_dataset_type)
def global_next_fn(server_weights, federated_dataset):

  # Each client computes their updated weights.
  client_weights = tff.federated_map(
      global_client_update_fn, federated_dataset)

  # The server averages these updates.
  mean_client_weights = tff.federated_mean(client_weights)

  # The server updates its model.
  server_weights = tff.federated_map(server_update_fn, mean_client_weights)

  return server_weights

global_federated_algorithm = tff.templates.IterativeProcess(
    initialize_fn=global_initialize_fn,
    next_fn=global_next_fn
)

#list of server states of local federations
local_server_states=[]

#intialize local servers by global server state value

for i in range(0,number_of_federation):
  local_server_states.append(global_server_state)

'''implementing multi tier federated learning'''

num_round=200

for i in range(0,num_round):
  #run a round of federated learnig for all local federation(update the local servers)
  for k in range(0, number_of_federation):
    local_server_states[k]=list_of_tff_tmpl[k].next(local_server_states[k],multi_tier_fed_train_dataset[k])
  print(f"round {i}")
  #update global server using local server states(mean of local servers)
  global_server_state = global_federated_algorithm.next(global_server_state, local_server_states)
  server_updates=local_server_states
  #update local servers using global states
  local_server_states[:len(local_server_states)]=[global_server_state]*len(local_server_states)

#save model

import datetime

def model_save(model):
    # Generate a unique identifier using the current date and time
     timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
     
     # Define the directory where you want to save the model
     save_path = r'/var/share/rs1/projects/Fed_STLF/federated_models/model_{}'.format(timestamp)
     
     # Save the model with the unique identifier
     model.save(save_path)


# craete a model from weights
model_evaluation=create_uncompiled_model_CONVLSTM(G.WINDOW_SIZE)
window_size=G.WINDOW_SIZE

#set the model weights wiht global server weights 
model_evaluation.set_weights(local_server_states[0])
model_save(model_evaluation)

import matplotlib.pyplot as plt

'''evalution'''

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

def compute_metrics(true_series, forecast):
    
    ''' first section  the errors based on absolute error'''
    
    
    mae=np.mean(np.abs(true_series-forecast))
       
    #Compute Relative Absolute Error(RAE), can only range from zero to one
    # true_mean=np.mean(true_series)
    # abs_error_sum=np.sum(np.abs(true_series-forecast))
    # true_deviation_sum=np.sum(np.abs(true_series-true_mean))
    # rae=abs_error_sum/true_deviation_sum
    numerator = np.sum(np.abs(forecast - true_series))

    denominator = np.sum(np.abs(np.mean(true_series) - true_series))
    rae=numerator / denominator
    
    #compute Mean Absolute Percetage Error(MAPE)
    # abs_error=(np.abs(true_series-forecast))/true_series
    # mape=np.mean(abs_error)*100
    mape=np.mean(np.abs((true_series - forecast) / true_series)) * 100
  
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

series=clustered_multi_federated_data[0][0]

print(series)
split=int(G.SPLIT_TIME*len(series))
model_performance_visulization(model_evaluation,series,scaler,window_size,split)



