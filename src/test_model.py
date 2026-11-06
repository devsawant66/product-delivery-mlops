import os
import joblib
import numpy as np
import pandas as pd


def test_model_file_exists():
    assert os.path.exists("delivery_time_model.pkl"), "Model file not found!"


def test_model_can_load():
    model = joblib.load("delivery_time_model.pkl")
    assert model is not None, "Model failed to load"


def test_model_prediction():
    model = joblib.load("delivery_time_model.pkl")

    sample_input = pd.DataFrame([{
        "Weather": "Sunny",
        "Traffic_Level": "Low",
        "Time_of_Day": "Morning",
        "Vehicle_Type": "Bike",
        "Distance_km": 10,
        "Preparation_Time_min": 5,
        "Courier_Experience_yrs": 3
    }])

    prediction = model.predict(sample_input)

    assert prediction is not None
    assert len(prediction) == 1
