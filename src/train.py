import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load CSV data
df = pd.read_csv("data/delivery_data.csv")

# Drop ID column
df = df.drop(columns=["Order_ID"])

# Features and target
X = df.drop(columns=["Delivery_Time_min"])
y = df["Delivery_Time_min"]

# Categorical and numerical columns
categorical_cols = [
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type"
]

numerical_cols = [
    "Distance_km",
    "Preparation_Time_min",
    "Courier_Experience_yrs"
]



# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

# Model
model = RandomForestRegressor(random_state=42)

# Pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
pipeline.fit(X_train, y_train)

# =========================
# BASELINE EVALUATION
# =========================

# Baseline: always predict mean delivery time
baseline_prediction = y_train.mean()
baseline_preds = [baseline_prediction] * len(y_test)

baseline_mae = mean_absolute_error(y_test, baseline_preds)

# =========================
# MODEL EVALUATION
# =========================

model_preds = pipeline.predict(X_test)
model_mae = mean_absolute_error(y_test, model_preds)

print("\n--- Model Evaluation ---")
print(f"Baseline MAE (Mean Predictor): {baseline_mae:.2f} minutes")
print(f"Model MAE (RandomForest): {model_mae:.2f} minutes")

# =========================
# QUALITY GATE (CI CHECK)
# =========================

MAX_ALLOWED_MAE = 10.0

if model_mae > MAX_ALLOWED_MAE:
    raise ValueError(
        f"Model MAE {model_mae:.2f} exceeds allowed threshold {MAX_ALLOWED_MAE}"
    )
else:
    print("Model quality check PASSED")


improvement = baseline_mae - model_mae
print(f"Improvement over baseline: {improvement:.2f} minutes")


# Save model
joblib.dump(pipeline, "delivery_time_model.pkl")
print("Model saved successfully")
