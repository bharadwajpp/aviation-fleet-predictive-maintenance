# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

app = FastAPI(title="Aviation RUL Prediction API")

# -------------------------
# Load pre-trained model
# -------------------------
MODEL_PATH = Path("models/rul_predictor_fd001.pkl")

if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
else:
    model = None
    print("WARNING: Model file not found. API will return dummy predictions.")

FEATURE_COLUMNS = [f"s{i}" for i in range(1, 22)] + ["time_in_cycles"]

# -------------------------
# Request Schema
# -------------------------
class EngineRequest(BaseModel):
    unit_number: int
    features: list  # must match FEATURE_COLUMNS length

# -------------------------
# Predict RUL endpoint
# -------------------------
@app.post("/predict_rul")
def predict_rul(req: EngineRequest):
    if model is not None:
        df = pd.DataFrame([req.features], columns=FEATURE_COLUMNS)
        prediction = model.predict(df)
        predicted_rul = float(prediction[0])
    else:
        # Dummy fallback
        predicted_rul = max(0, 100 - req.unit_number)

    return {"unit_number": req.unit_number, "predicted_RUL": predicted_rul}
