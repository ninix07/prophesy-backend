from numpy import random
import numpy as np
from tensorflow import keras
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

from model_training.model_training import ModelTraining


def model_prediction(x_log):
    prediciton_model = ModelTraining.getInstance().Battery_model
    prediction_scaler = ModelTraining.getInstance().output_scaler
    # load the model and output the prediction
    y_log_pred = prediciton_model.predict(x_log)
    return prediction_scaler.inverse_transform(y_log_pred)
