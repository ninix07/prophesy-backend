from numpy import random
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from model_training.model_training import ModelTraining
from config import DATA_CHUNKS,RESOURCES_FOLDER_NAME,LINEAR_REGRESSION_MODEL_FOLDER_NAME,LINEAR_REGRESSION_MODEL_NAME
import pickle
from pathlib import Path

def checkfilerequirements():
    MODEL_TRAINING_FILE_PATH = Path(__file__)
    RESOURCESS_PATH = MODEL_TRAINING_FILE_PATH.parents[1].joinpath(
                    RESOURCES_FOLDER_NAME)
    LINEAR_MODEL_FILE_PATH=RESOURCESS_PATH.joinpath(LINEAR_REGRESSION_MODEL_FOLDER_NAME)
    LINEAR_MODEL_PATH= LINEAR_MODEL_FILE_PATH.joinpath(LINEAR_REGRESSION_MODEL_NAME)
    return LINEAR_MODEL_PATH


def model_prediction(x_log):
    ## X-Log data : Voltage,Energy,Energy_Rate[J/s],State,time_diff(s)
    prediciton_model = ModelTraining.getInstance().Battery_model
    data_index=ModelTraining.getInstance().data_index
    chunk_limit=3*DATA_CHUNKS
    if(data_index < chunk_limit):
        print("Entered")
        data=[]
        for row in x_log:
            energy_rate=(-(row[2]/3.6) if row[4]==-1 else (row/3.6))
            print(energy_rate)
            time_diff=row[3]
            data.append([energy_rate , time_diff])
        model_filename=checkfilerequirements()
        print(model_filename)
        model=pickle.load(open(model_filename,'rb'))
        y=model.predict(data)
        y=y*3.6
        return y
    
    prediction_scaler = ModelTraining.getInstance().output_scaler
    # load the model and output the prediction
    y_log_pred = prediciton_model.predict(x_log)
    return prediction_scaler.inverse_transform(y_log_pred)

