import joblib
import json
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "outputs/models/rul_predictor.pkl"
SCALER_PATH = BASE_DIR / "outputs/models/scaler.pkl"
FEATURES_PATH = BASE_DIR / "outputs/models/feature_names.json"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(FEATURES_PATH, "r") as f:
    FEATURE_NAMES = json.load(f)

def predict_rul(features: dict):
    x = np.array([features[f] for f in FEATURE_NAMES]).reshape(1, -1)
    x_scaled = scaler.transform(x)
    rul = model.predict(x_scaled)[0]
    return max(0, float(rul))
