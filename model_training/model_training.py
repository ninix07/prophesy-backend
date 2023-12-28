import csv
from config import BATTERY_LOG_PATH, POLLING_INTERVAL
import time
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import load_model


def model_training():
    # Testing polling in csv reading
    # Need to do online training for model
    
    # Checking headers in csv if not adding a headers
    csv = pd.read_csv(BATTERY_LOG_PATH)
    headerList = ["Date","Battery Index","Battery State","Voltage[V]","Energy[J]", "Energy_Rate[J/s]"]
    csv.to_csv(BATTERY_LOG_PATH, header=headerList, index=False)

    while True:
        # CSV polling for reading
        time.sleep(POLLING_INTERVAL)
