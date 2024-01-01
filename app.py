import time
from flask import Flask,request,jsonify
from threading import Thread
from config import POLLING_INTERVAL
from model_training.model_training import model_training
from model_prediction.model_prediction import model_prediction

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "hello world"


@app.route("/predict",methods = ["GET"])
def predict():
    # Serializing query into list of list of integres
    query_string = request.args.get("x_log")
    inner_string = query_string[2:-2]
    inner_list = inner_string.split(',')
    converted_list = [int(float(num)) if '.' not in num else float(num) for num in inner_list]
    query =[converted_list]
    
    result =  model_prediction(query)
    
    print(type(result))
    # Converting np array into list and jsonifying it 
    return jsonify(array=result.tolist())


if __name__ == '__main__':
    # as it is not the main thread
    # ie. running in background
    # So, daemon thread for handling keyboard interrupts
    p = Thread(target=model_training, daemon=True)
    p.start()
    app.run(debug=True)
