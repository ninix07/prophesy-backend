from tensorflow import keras 
from keras.models import load_model

model = load_model("Battery_model")

def model_prediction(x_log):
    # load the model and output the prediction
    y_log_pred = model.predict(x_log)
    return y_log_pred
