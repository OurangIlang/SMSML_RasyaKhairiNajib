from fastapi import FastAPI
from pydantic import BaseModel
from inference import predict

app = FastAPI(title="Medical Insurance Prediction API")

class InsuranceInput(BaseModel):
    age: int
    bmi: float
    children: int
    sex: str
    smoker: str
    region: str

@app.get("/")
def root():
    return {"status": "API is running"}

@app.post("/predict")
def predict_insurance(data: InsuranceInput):
    result = predict(data.dict())
    return {
        "prediction": float(result[0])
    }