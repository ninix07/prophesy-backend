import os
from pathlib import Path
import sys
from config import BATTERY_LOG_PATH, DATA_CHUNKS, POLLING_INTERVAL
import time
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping


from sklearn.preprocessing import MinMaxScaler

data_index= 0

if os.listdir("./").__contains__("Battery_model"):
    Battery_model = load_model("Battery_model")
    output_scaler= np.load("../Battery_model/Output_scaler.npy")
    with open('../Battery_model/dataIndex.txt', 'r') as f:
        data_index = int(f.read())
else:
    Battery_model = tf.keras.Sequential()
    Battery_model.add(tf.keras.layers.BatchNormalization())
    Battery_model.add(tf.keras.layers.Dense(32, activation="tanh"))

    Battery_model.add(tf.keras.layers.BatchNormalization())
    Battery_model.add(tf.keras.layers.Dense(128, activation="tanh"))

    Battery_model.add(tf.keras.layers.BatchNormalization())
    Battery_model.add(tf.keras.layers.Dense(32, activation="tanh"))
    Battery_model.add(tf.keras.layers.Dense(
        1, activation='linear', trainable=True))

    Battery_model.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True)

output_scaler = MinMaxScaler(feature_range=(-1, 1))


def model_training():
    # Checking headers in csv if not adding a headers
    log_data = pd.read_csv(BATTERY_LOG_PATH)
    headerList = ["Date", "Battery Index", "Battery State", "Voltage[V]",
                  "Energy[J]", "Energy_Rate[J/s]", 'Time Difference[s]', 'Capacity Difference[J]']
    log_data.columns = headerList
    log_data.to_csv(BATTERY_LOG_PATH, index=False)
    
    while True:
        # Importing data in chunks of 1000
        file_chunks = pd.read_csv(BATTERY_LOG_PATH, skiprows=range(0,data_index), chunksize=DATA_CHUNKS)
        while True:
            try:
               model_fitting(datas=next(file_chunks))
            except:
                time.sleep(POLLING_INTERVAL)


def model_fitting(datas):
    X_log = datas[['Voltage[V]', 'Energy[J]','Energy_Rate[J/s]', 'Time Difference[s]', 'Battery State']]
    y_log = datas['Capacity Difference[J]']
    capacity_difference_column = y_log.values.reshape(-1, 1)
    Y_normalized = output_scaler.fit_transform(
        capacity_difference_column)
    output_scaler.inverse_transform(Y_normalized)
    Battery_model.fit(X_log, Y_normalized, epochs=50, batch_size=5,
                        validation_split=0.2, callbacks=[early_stopping])
    Battery_model.save("Battery_model")
    np.save("../Battery_model/Output_scaler.npy",output_scaler)
    data_index+=datas.shape[1]
    with open('../Battery_model/dataIndex.txt', 'w') as f:
        f.write(data_index+DATA_CHUNKS)
                