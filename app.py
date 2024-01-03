import time
from flask import Flask,request,jsonify
from threading import Thread
from config import POLLING_INTERVAL
from model_training.model_training import ModelTraining
from model_prediction.model_prediction import model_prediction
from linear_regression.linear_regression import linear_regression_train
import ast
from config import DATA_CHUNKS
app = Flask(__name__)


@app.route("/")
def hello_world():
    return "hello world"


@app.route("/predict",methods = ["GET"])
def predict():
    # Serializing query into list of list of floats
    query =ast.literal_eval(request.args.get("x_log"))
    
    result =  model_prediction(query)
    
    print(type(result))
    # Converting np array into list and jsonifying it 
    return jsonify(array=result.tolist())


if __name__ == '__main__':
    # as it is not the main thread
    # ie. running in background
    # So, daemon thread for handling keyboard interrupts
  
    p = Thread(target=ModelTraining.getInstance().model_training, daemon=True)
    p.start()
    if(ModelTraining.getInstance().data_index < (3*DATA_CHUNKS)):
        q=Thread(target=linear_regression_train,daemon=True)
        q.start()
    
    app.run(debug=False)
