import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import load_model
import tensorflow as tf
from keras.callbacks import EarlyStopping
from threading import Thread

from sklearn.preprocessing import MinMaxScaler

from config import BATTERY_LOG_PATH, DATA_CHUNKS, POLLING_INTERVAL, MODEL_FOLDER_NAME, RESOURCES_FOLDER_NAME, OUTPUT_SCALER_NAME, DATA_INDEX_FILE_NAME
from pathlib import Path


MODEL_TRAINING_FILE_PATH = Path(__file__)
RESOURCESS_PATH = MODEL_TRAINING_FILE_PATH.parents[1].joinpath(RESOURCES_FOLDER_NAME)

BATTERY_MODEL_PATH = RESOURCESS_PATH.joinpath(MODEL_FOLDER_NAME)
MINMAX_SCALAR_PATH = RESOURCESS_PATH.joinpath(OUTPUT_SCALER_NAME)
DATA_INDEX_FILE_PATH = RESOURCESS_PATH.joinpath(DATA_INDEX_FILE_NAME)

if not os.path.exists(RESOURCESS_PATH):
    print(f"The path {RESOURCESS_PATH} doesn't exits so creating it.")
    os.mkdir(RESOURCESS_PATH)

data_index = 0

if os.path.exists(BATTERY_MODEL_PATH):
    Battery_model = load_model(BATTERY_MODEL_PATH)
    loaded_scaler = np.load(MINMAX_SCALAR_PATH, allow_pickle=True)
    output_scaler= loaded_scaler.item()
    with open(DATA_INDEX_FILE_PATH, 'r') as f:
        data_index = int(f.read())
    model_previously_present = True
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
        chunksize = DATA_CHUNKS
        data_shape_1 = log_data.shape[0]
        if data_shape_1 < DATA_CHUNKS and data_index == 0:
            chunksize = data_shape_1

        if data_index != 0 and (data_shape_1 - data_index) < DATA_CHUNKS: 
            time.sleep(POLLING_INTERVAL)
            continue

        file_chunks = pd.read_csv(BATTERY_LOG_PATH, skiprows=range(0,data_index), chunksize=chunksize)
        # 
        for chunk in file_chunks:
            try:
                model_fitting(datas=chunk)
            except Exception as e:
                time.sleep(POLLING_INTERVAL)

from model_prediction import update_model_and_scaler

def model_fitting(datas):
    global data_index, output_scaler

    X_log = datas[['Voltage[V]', 'Energy[J]','Energy_Rate[J/s]', 'Time Difference[s]', 'Battery State']]
    y_log = datas['Capacity Difference[J]']
    capacity_difference_column = y_log.values.reshape(-1, 1)

    Y_normalized = output_scaler.fit_transform(capacity_difference_column)
    
    # output_scaler.inverse_transform(Y_normalized)

    Battery_model.fit(X_log, Y_normalized, epochs=50, batch_size=5,
                        validation_split=0.2, callbacks=[early_stopping])
    # do this on separate thread
    p = Thread(target=update_model_and_scaler, args=[Battery_model, output_scaler], daemon=True)
    p.start()

    Battery_model.save(BATTERY_MODEL_PATH)
    np.save(MINMAX_SCALAR_PATH,output_scaler, allow_pickle=True)
    data_index += datas.shape[0]

    with open(DATA_INDEX_FILE_PATH, 'w') as f:
        f.write(str(data_index))
