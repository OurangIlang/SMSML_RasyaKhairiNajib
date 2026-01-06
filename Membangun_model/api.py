import os
import time
import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://host.docker.internal:5000" 
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_URI = "models:/medical_insurance_model@production"
model = mlflow.pyfunc.load_model(MODEL_URI)

app = FastAPI(title="Medical Insurance Prediction API")

REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests"
)

REQUEST_LATENCY = Histogram(
    "prediction_request_latency_seconds",
    "Prediction latency"
)

class InputData(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

@app.post("/predict")
def predict(data: InputData):
    start = time.time()
    df = pd.DataFrame([data.dict()])
    pred = model.predict(df)

    REQUEST_COUNT.inc()
    REQUEST_LATENCY.observe(time.time() - start)

    return {"log_prediction": float(pred[0])}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
