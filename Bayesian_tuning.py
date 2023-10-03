# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from helper import *
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.config.list_physical_devices('GPU')
class config:
    # define the path to our output directory
    OUTPUT_PATH = "output"
    
    # define the total number of epochs to train, batch size, and the
    # early stopping patience
    EPOCHS = 50
    BS = 32
    EARLY_STOPPING_PATIENCE = 5

''' load, normalize data and create dataset using keras.data.dataset clss'''
# Specify the path to the CSV file
csv_file_path = r"/var/share/rs1/LCL_DATA/preparded_houshold_data/block0_MAC000002.csv"

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))



# Define window sizes to search
window_sizes = [8, 10,16, 12,24, 48]

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
    batch_size = 32
    train_dataset = train_dataset.shuffle(len(X_train)).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)
    
    return train_dataset, test_dataset


#read the csv file
df=pd.read_csv(csv_file_path)
#craete series
series=df['energy(kWh/hh)']

# Reshape the data to a 2D array as required by the scaler
series = series.values.reshape(-1, 1)

#fit scaler
norm_series = scaler.fit_transform(series)


window_size=G.WINDOW_SIZE
train_dataset, test_dataset=create_dataset(norm_series,window_size)

'''end train dataset creation'''

# Define the objective function for hyperparameter tuning
def objective(hp):
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
    #                   input_shape=[None]),
    #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hp.Int('units_1', min_value=64, max_value=256, step=32), return_sequences=True)),
    #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hp.Int('units_2', min_value=64, max_value=256, step=32))),
    #     tf.keras.layers.Dense(hp.Int('units_3', min_value=16, max_value=64, step=16)),
    #     tf.keras.layers.Dense(1)
    # ])
    # model = tf.keras.models.Sequential([
    # tf.keras.layers.Conv1D(hp.Int('conv_1', min_value=32, max_value=96, step=32), kernel_size=3,
    #                   strides=1,
    #                   activation="relu",
    #                   padding='causal',
    #                   input_shape=[G.WINDOW_SIZE, 1]),
    # tf.keras.layers.LSTM(hp.Int('units_1', min_value=64, max_value=256, step=32), return_sequences=True),
    # tf.keras.layers.LSTM(hp.Int('units_2', min_value=64, max_value=256, step=32)),
    # tf.keras.layers.Dense(hp.Int('units_3', min_value=16, max_value=64, step=16), activation="relu"),
    # tf.keras.layers.Dense(hp.Int('units_3', min_value=8, max_value=32, step=8), activation="relu"),
    # tf.keras.layers.Dense(1)])
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=hp.Int('conv_1', min_value=32, max_value=96, step=32), kernel_size=3,
                      strides=1,
                      activation="relu",
                      padding='causal',
                      input_shape=[window_size, 1]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hp.Int('Bi_LSTM_1', min_value=64, max_value=256, step=32), return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hp.Int('Bi_LSTM_2', min_value=64, max_value=256, step=32))),
    tf.keras.layers.Dense(hp.Int('Dense_1', min_value=16, max_value=64, step=16), activation="relu"),
    tf.keras.layers.Dense(hp.Int('Dense_2', min_value=8, max_value=32, step=8), activation="relu"),
    tf.keras.layers.Dense(1)])
    
    lr=hp.Choice('learning_rate', values=[1e-2,5e-2 ,1e-3, 5e-3,1e-4,5e-4,1e-5])
    loss_choice=hp.Choice('loss',values=['mse','mae'])
    optimizer=tf.keras.optimizers.SGD(momentum=0.9,learning_rate = lr)

    model.compile(loss=loss_choice,
                  optimizer=optimizer,metrics=['mse','mae'])

    
    
    return model

# initialize an early stopping callback to prevent the model from
# overfitting/spending too much time training with minimal gains
es = EarlyStopping(monitor="val_loss",
                   patience=config.EARLY_STOPPING_PATIENCE,
                   restore_best_weights=True)

# Instantiate the tuner
tuner = kt.BayesianOptimization(objective,objective="val_accuracy",
                                max_trials=10,
                                seed=42, 
                                directory=config.OUTPUT_PATH, 
                                project_name='hyperparameter_tuning')




#perform the hyperparameter search
print("[INFO] performing hyperparameter search...")
tuner.search(
 	x=train_dataset,
    validation_data=test_dataset,
    batch_size=config.BS,
 	callbacks=[es],
 	epochs=config.EPOCHS
)

#grab the best hyperparameters
bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
print("[INFO] optimal number of filters in conv_1 layer: {}".format(
 	bestHP.get('conv_1')))
print("[INFO] optimal number of units in Bi_LSTM_1 layer: {}".format(
 	bestHP.get('Bi_LSTM_1')))
print("[INFO] optimal number of units in Bi_LSTM_2 layer: {}".format(
 	bestHP.get('Bi_LSTM_2')))
print("[INFO] optimal number of units in Dense_1 layer: {}".format(
 	bestHP.get('Dense_1')))
print("[INFO] optimal number of units in Dense_2 layer: {}".format(
 	bestHP.get('Dense_2')))
print("[INFO] optimal learning rate: {:.4f}".format(
 	bestHP.get('learning_rate')))
print("[INFO] optimal loss function: {}".format(
 	bestHP.get('loss')))


def save_plot(H, path):
	# plot the training loss and accuracy
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["loss"], label="train_loss")
	plt.plot(H.history["val_loss"], label="val_loss")
	plt.plot(H.history["accuracy"], label="train_acc")
	plt.plot(H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")

















