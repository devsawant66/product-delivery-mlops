from fastapi import FastAPI
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from predict import predict_delivery_time

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Delivery Time ML API Running on Azure"}

@app.get("/predict")
def predict():
    return predict_delivery_time()