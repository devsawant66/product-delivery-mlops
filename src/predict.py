import joblib
import pandas as pd

# Load trained model
model = joblib.load("delivery_time_model.pkl")

print("\n--- Enter Product Delivery Details ---")

distance = float(input("Enter distance (km): "))

print("\nSelect Traffic Level:")
print("1. Low")
print("2. Medium")
print("3. High")
traffic_choice = input("Enter choice (1/2/3): ")
traffic_map = {"1": "Low", "2": "Medium", "3": "High"}
traffic = traffic_map.get(traffic_choice, "Medium")

print("\nSelect Weather:")
print("1. Clear")
print("2. Rainy")
print("3. Foggy")
weather_choice = input("Enter choice (1/2/3): ")
weather_map = {"1": "Clear", "2": "Rainy", "3": "Foggy"}
weather = weather_map.get(weather_choice, "Clear")

print("\nSelect Time of Day:")
print("1. Morning")
print("2. Afternoon")
print("3. Evening")
print("4. Night")
time_choice = input("Enter choice (1/2/3/4): ")
time_map = {
    "1": "Morning",
    "2": "Afternoon",
    "3": "Evening",
    "4": "Night"
}
time_of_day = time_map.get(time_choice, "Evening")

print("\nSelect Vehicle Type:")
print("1. Bike")
print("2. Scooter")
print("3. Car")
vehicle_choice = input("Enter choice (1/2/3): ")
vehicle_map = {"1": "Bike", "2": "Scooter", "3": "Car"}
vehicle = vehicle_map.get(vehicle_choice, "Bike")

processing_time = float(
    input("\nEnter order processing time at warehouse (minutes): ")
)

experience = float(
    input("Enter courier experience (years): ")
)

# Prepare input for prediction
user_input = pd.DataFrame([{
    "Distance_km": distance,
    "Weather": weather,
    "Traffic_Level": traffic,
    "Time_of_Day": time_of_day,
    "Vehicle_Type": vehicle,
    "Preparation_Time_min": processing_time,

    "Courier_Experience_yrs": experience
}])

# Predict
prediction = model.predict(user_input)

print("\n====================================")
print(f"Estimated Delivery Time: {prediction[0]:.2f} minutes")
print("====================================\n")
