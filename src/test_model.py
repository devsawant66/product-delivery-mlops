import os
import joblib
import numpy as np


def test_model_file_exists():
    assert os.path.exists("delivery_time_model.pkl"), "Model file not found!"


def test_model_can_load():
    model = joblib.load("delivery_time_model.pkl")
    assert model is not None, "Model failed to load"


def test_model_prediction():
    model = joblib.load("delivery_time_model.pkl")

    # Adjust number of features if needed
    sample_input = np.array([[10, 5, 3]])

    prediction = model.predict(sample_input)

    assert prediction is not None
    assert len(prediction) == 1
