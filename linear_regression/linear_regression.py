from config import BATTERY_HISTORY_PATH, RESOURCES_FOLDER_NAME,LINEAR_REGRESSION_MODEL_FOLDER_NAME,LINEAR_REGRESSION_MODEL_NAME
import numpy as np
import pandas as pd
from sklearn import linear_model
import pickle
from pathlib import Path
import os

def check_and_create_file_path():
    MODEL_TRAINING_FILE_PATH = Path(__file__)
    RESOURCESS_PATH = MODEL_TRAINING_FILE_PATH.parents[1].joinpath(
                    RESOURCES_FOLDER_NAME)
    LINEAR_MODEL_FILE_PATH=RESOURCESS_PATH.joinpath(LINEAR_REGRESSION_MODEL_FOLDER_NAME)
    if not os.path.exists(RESOURCESS_PATH):
                print(f"The path {RESOURCESS_PATH} doesn't exits so creating it.")
                os.mkdir(RESOURCESS_PATH)
    if not os.path.exists(LINEAR_MODEL_FILE_PATH):
                os.mkdir(LINEAR_MODEL_FILE_PATH)
    return LINEAR_MODEL_FILE_PATH.joinpath(LINEAR_REGRESSION_MODEL_NAME)

def data_clean(data):
    data['date_time'] = pd.to_datetime(data['date_time'], unit='s')
    data=data.sort_values(by=['date_time'],ascending=True)
    data_slope = pd.DataFrame(columns=['prev_time', 'curr_time', 'prev_capacity', 'curr_capacity'])
    data_slope['prev_time'] = data['date_time'].shift()
    data_slope['curr_time'] = data['date_time']
    data_slope['prev_capacity'] = data['capacity'].shift()
    data_slope['curr_capacity'] = data['capacity']
    data_slope['time_diff']=(data_slope['curr_time'] - data_slope['prev_time']).dt.total_seconds()
    data_slope['capacity_diff']=(data_slope['curr_capacity'] - data_slope['prev_capacity'])
    data_slope['slope'] = (data_slope['curr_capacity'] - data_slope['prev_capacity']) / data_slope['time_diff']
    data_slope=data_slope.dropna()
    return data_slope


def linear_regression_train():
    data= pd.read_csv(BATTERY_HISTORY_PATH)
    data=data_clean(data)
    linear_regression_model = linear_model.LinearRegression()
    X = data[['slope', 'time_diff']]
    y=data['capacity_diff']
    linear_regression_model.fit(X, y)
    print("Successfully fit Linear Regression data")
    model_filename = check_and_create_file_path()
    with open(model_filename, 'wb') as model_file:
     pickle.dump(linear_regression_model, model_file)