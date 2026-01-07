"""
API to serve trained RUL model predictions
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import json
from pathlib import Path
from typing import List, Optional

app = FastAPI(
    title="Aviation Fleet RUL Prediction API",
    version="1.0",
    description="Serve trained Random Forest model for aircraft engine RUL predictions"
)

# Load artifacts
MODEL_PATH = Path("outputs/models/rul_predictor.pkl")
SCALER_PATH = Path("outputs/models/scaler.pkl")
FEATURE_PATH = Path("outputs/models/feature_names.json")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(FEATURE_PATH, 'r') as f:
    feature_names = json.load(f)

# Request body
class EngineData(BaseModel):
    engine_id: int
    cycle: int
    sensors: dict  # sensor_name: value

class FleetData(BaseModel):
    engines: List[EngineData]

@app.get("/")
def root():
    return {"status": "RUL Prediction API Online"}

@app.post("/predict")
def predict_rul(data: FleetData):
    df_list = []
    for engine in data.engines:
        row = {"engine_id": engine.engine_id, "cycle": engine.cycle}
        for feature in feature_names:
            row[feature] = engine.sensors.get(feature, 0.0)
        df_list.append(row)
    
    df = pd.DataFrame(df_list)
    
    # Scale
    X = scaler.transform(df[feature_names])
    
    # Predict
    preds = model.predict(X)
    
    df["predicted_RUL"] = preds
    df["risk_level"] = pd.cut(
        df["predicted_RUL"],
        bins=[0,30,70,float('inf')],
        labels=['Critical','Warning','Normal']
    )
    
    results = df.to_dict(orient="records")
    return {"predictions": results}
