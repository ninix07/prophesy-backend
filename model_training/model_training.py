import csv
from config import BATTERY_LOG_PATH, POLLING_INTERVAL
import time


def model_training():
    # Testing polling in csv reading
    # Need to do online training for model
    i = 0
    while True:
        i += 1
        with open("abc.txt", "w") as file:
            file.write(i.__str__())
        time.sleep(POLLING_INTERVAL)
