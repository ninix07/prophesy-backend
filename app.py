import time
from flask import Flask
from threading import Thread
from config import POLLING_INTERVAL
from model_training.model_training import model_training
from model_prediction.model_prediction import model_prediction

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "hello world"


@app.route("/pedict")
def predict():
    return model_prediction().__str__()


if __name__ == '__main__':
    # as it is not the main thread
    # ie. running in background
    # So, daemon thread for handling keyboard interrupts
    p = Thread(target=model_training, daemon=True)
    p.start()
    app.run()
