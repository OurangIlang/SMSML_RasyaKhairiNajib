from fastapi import FastAPI
from pydantic import BaseModel
from inference import predict
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Medical Insurance API")


Instrumentator().instrument(app).expose(app)

class InsuranceInput(BaseModel):
    age: int
    bmi: float
    children: int
    sex: str
    smoker: str
    region: str

@app.post("/predict")
def predict_insurance(data: InsuranceInput):
    result = predict(data.dict())
    return {"prediction": float(result[0])}
