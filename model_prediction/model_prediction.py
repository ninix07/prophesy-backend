from numpy  import random
import numpy as np
from tensorflow import keras 
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from model_training.model_training import Battery_model,output_scaler

def model_prediction(x_log):
    # load the model and output the prediction
    y_log_pred = Battery_model.predict(x_log)
    return output_scaler.inverse_transform(y_log_pred)
    
    

