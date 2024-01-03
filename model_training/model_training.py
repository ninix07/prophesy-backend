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


class ModelTraining:
    __instance = None
   

    @staticmethod
    def getInstance():
        if ModelTraining.__instance == None:
            ModelTraining()
        return ModelTraining.__instance

    def __init__(self):
        if ModelTraining.__instance != None:
            raise Exception(
                "Object already created please use get_instance method")
        else:
            ModelTraining.__instance = self
            self.__MODEL_TRAINING_FILE_PATH = Path(__file__)
            self.__RESOURCESS_PATH = self.__MODEL_TRAINING_FILE_PATH.parents[1].joinpath(
                RESOURCES_FOLDER_NAME)

            self.__BATTERY_MODEL_PATH = self.__RESOURCESS_PATH.joinpath(MODEL_FOLDER_NAME)
            self.__MINMAX_SCALAR_PATH = self.__RESOURCESS_PATH.joinpath(OUTPUT_SCALER_NAME)
            self.__DATA_INDEX_FILE_PATH = self.__RESOURCESS_PATH.joinpath(DATA_INDEX_FILE_NAME)
            
            self.data_index = 0
            if not os.path.exists(self.__RESOURCESS_PATH):
                print(f"The path {self.__RESOURCESS_PATH} doesn't exits so creating it.")
                os.mkdir(self.__RESOURCESS_PATH)
            if os.path.exists(self.__BATTERY_MODEL_PATH):
                self.Battery_model = load_model(self.__BATTERY_MODEL_PATH)
                loaded_scaler = np.load(self.__MINMAX_SCALAR_PATH, allow_pickle=True)
                self.output_scaler = loaded_scaler.item()
                with open(self.__DATA_INDEX_FILE_PATH, 'r') as f:
                    self.data_index = int(f.read())
            else:
                self.Battery_model = tf.keras.Sequential()
                self.Battery_model.add(tf.keras.layers.BatchNormalization())
                self.Battery_model.add(tf.keras.layers.Dense(32, activation="tanh"))
                self.Battery_model.add(tf.keras.layers.BatchNormalization())
                self.Battery_model.add(tf.keras.layers.Dense(128, activation="tanh"))
                self.Battery_model.add(tf.keras.layers.BatchNormalization())
                self.Battery_model.add(tf.keras.layers.Dense(32, activation="tanh"))
                self.Battery_model.add(tf.keras.layers.Dense(
                    1, activation='linear', trainable=True))
                self.Battery_model.compile(optimizer='adam', loss='mean_squared_error')
                self.__early_stopping = EarlyStopping(
                    monitor='val_loss', patience=5, restore_best_weights=True)
                self.output_scaler = MinMaxScaler(feature_range=(-1, 1))
    def model_training(self):
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
            if data_shape_1 < DATA_CHUNKS and self.data_index == 0:
                chunksize = data_shape_1

            if self.data_index != 0 and (data_shape_1 - self.data_index) < DATA_CHUNKS:
                time.sleep(POLLING_INTERVAL)
                continue

            file_chunks = pd.read_csv(BATTERY_LOG_PATH, skiprows=range(
                0, self.data_index), chunksize=chunksize)
            #
            for chunk in file_chunks:
                chunk.columns = headerList
                try:
                    self.__model_fitting(datas=chunk)
                except Exception as e:
                    print(e)
                    time.sleep(POLLING_INTERVAL)


    def __model_fitting(self,datas):
        X_log = datas[['Voltage[V]', 'Energy[J]', 'Energy_Rate[J/s]',
                    'Time Difference[s]', 'Battery State']]
        y_log = datas['Capacity Difference[J]']
        capacity_difference_column = y_log.values.reshape(-1, 1)

        Y_normalized = self.output_scaler.fit_transform(capacity_difference_column)

        # output_scaler.inverse_transform(Y_normalized)

        self.Battery_model.fit(X_log, Y_normalized, epochs=50, batch_size=5,
                        validation_split=0.2, callbacks=[self.__early_stopping])
        # # do this on separate thread
        # p = Thread(target=update_model_and_scaler, args=[
        #         self.Battery_model, self.output_scaler], daemon=True)
        # p.start()

        self.Battery_model.save(self.__BATTERY_MODEL_PATH)
        np.save(self.__MINMAX_SCALAR_PATH, self.output_scaler, allow_pickle=True)
        self.data_index += datas.shape[0]

        with open(self.__DATA_INDEX_FILE_PATH, 'w') as f:
            f.write(str(self.data_index))
