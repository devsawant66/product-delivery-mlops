import mlflow
import mlflow.sklearn
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import torch
from datetime import datetime

# =========================
# DEVICE CHECK
# =========================
def check_device():
    if torch.cuda.is_available():
        device = "GPU"
    else:
        device = "CPU"
    print(f"Running training on: {device}")
    return device

device_used = check_device()

# =========================
# MLflow Setup
# =========================
mlflow.set_experiment("Delivery_Time_Project")

with mlflow.start_run():

    print("Training started...", flush=True)

    # =========================
    # LOAD DATA
    # =========================
    df = pd.read_csv("data/delivery_data.csv")
    df = df.drop(columns=["Order_ID"])

    X = df.drop(columns=["Delivery_Time_min"])
    y = df["Delivery_Time_min"]

    categorical_cols = [
        "Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"
    ]

    numerical_cols = [
        "Distance_km", "Preparation_Time_min", "Courier_Experience_yrs"
    ]

    # =========================
    # LOG PARAMETERS
    # =========================
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 10)
    mlflow.log_param("test_size", 0.2)

    # =========================
    # PREPROCESSING
    # =========================
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols)
        ]
    )

    model = RandomForestRegressor(n_estimators=10, random_state=42)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # =========================
    # SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Fitting model...", flush=True)
    pipeline.fit(X_train, y_train)
    print("Training completed.", flush=True)

    # =========================
    # BASELINE
    # =========================
    baseline_prediction = y_train.mean()
    baseline_preds = [baseline_prediction] * len(y_test)
    baseline_mae = mean_absolute_error(y_test, baseline_preds)

    # =========================
    # MODEL EVALUATION
    # =========================
    model_preds = pipeline.predict(X_test)
    model_mae = mean_absolute_error(y_test, model_preds)

    print("\n--- Model Evaluation ---")
    print(f"Baseline MAE: {baseline_mae:.2f}")
    print(f"Model MAE: {model_mae:.2f}")

    # =========================
    # LOG METRICS (MLflow)
    # =========================
    mlflow.log_metric("baseline_mae", baseline_mae)
    mlflow.log_metric("model_mae", model_mae)
    mlflow.log_metric("improvement", baseline_mae - model_mae)

    # =========================
    # QUALITY CHECK
    # =========================
    MAX_ALLOWED_MAE = 10.0

    if model_mae > MAX_ALLOWED_MAE:
        raise ValueError("Model failed quality check")
    else:
        print("Model quality check PASSED")

    # =========================
    # SAVE MODEL
    # =========================
    joblib.dump(pipeline, "delivery_time_model.pkl")

    # ALSO LOG MODEL IN MLFLOW
    mlflow.sklearn.log_model(pipeline, "model")

    # =========================
    # REGISTER MODEL (Experiment 8)
    # =========================
    run_id = mlflow.active_run().info.run_id

    mlflow.register_model(
        f"runs:/{run_id}/model",
        "DeliveryTimeModel"
    )

    # =========================
    # SAVE METRICS FILE
    # =========================
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "baseline_mae": round(baseline_mae, 2),
        "model_mae": round(model_mae, 2),
        "improvement": round(baseline_mae - model_mae, 2)
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Metrics saved successfully")