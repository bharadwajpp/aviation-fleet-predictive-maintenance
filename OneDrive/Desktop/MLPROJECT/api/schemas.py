from pydantic import BaseModel

# Keep this generalized for any aviation engine
class EngineSensorInput(BaseModel):
    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float
    sensor_22: float
    sensor_23: float
    sensor_24: float


class RULPredictionResponse(BaseModel):
    predicted_rul: float
    risk_level: str
