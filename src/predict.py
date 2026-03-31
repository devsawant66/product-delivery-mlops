import joblib
import pandas as pd
from datetime import datetime
import os

# Load model safely (important for Azure)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "delivery_time_model.pkl")

model = joblib.load(model_path)


def auto_generate_features():
    now = datetime.now()
    hour = now.hour

    if hour < 10:
        time_of_day = "Morning"
        traffic = "Low"
    elif hour < 17:
        time_of_day = "Afternoon"
        traffic = "High"
    elif hour < 21:
        time_of_day = "Evening"
        traffic = "High"
    else:
        time_of_day = "Night"
        traffic = "Medium"

    weather = "Clear"
    vehicle = "Bike" if traffic == "High" else "Car"
    order_processing_time = 15 if time_of_day in ["Afternoon", "Evening"] else 10
    courier_experience = 3
    distance_km = 12.0

    return {
        "Distance_km": distance_km,
        "Weather": weather,
        "Traffic_Level": traffic,
        "Time_of_Day": time_of_day,
        "Vehicle_Type": vehicle,
        "Preparation_Time_min": order_processing_time,
        "Courier_Experience_yrs": courier_experience
    }


def predict_delivery_time():
    features = auto_generate_features()
    input_df = pd.DataFrame([features])
    prediction = model.predict(input_df)

    return {
        "features_used": features,
        "predicted_delivery_time": round(float(prediction[0]), 2)
    }