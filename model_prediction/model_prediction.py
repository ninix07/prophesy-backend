from numpy import random
import numpy as np
from tensorflow import keras
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from model_training.model_training import Battery_model, output_scaler

prediciton_model = Battery_model
prediction_scaler = output_scaler


def model_prediction(x_log):
    # load the model and output the prediction
    y_log_pred = Battery_model.predict(x_log)
    return prediction_scaler.inverse_transform(y_log_pred)


def update_model(battery_model):
    global prediciton_model
    prediciton_model = battery_model


def update_scaler(output_scaler):
    global prediction_scaler
    prediction_scaler = output_scaler


def update_model_and_scaler(battery_model, output_scaler):
    update_model(battery_model)
    update_scaler(output_scaler)
