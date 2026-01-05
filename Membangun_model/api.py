from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from inference import predict

app = FastAPI(title="Medical Insurance API")

class InputData(BaseModel):
    age: int
    bmi: float
    children: int
    sex_male: int
    smoker_yes: int
    region_northwest: int
    region_southeast: int
    region_southwest: int

@app.post("/predict")
def predict_endpoint(data: InputData):
    input_array = np.array([[ 
        data.age,
        data.bmi,
        data.children,
        data.sex_male,
        data.smoker_yes,
        data.region_northwest,
        data.region_southeast,
        data.region_southwest
    ]])

    prediction = predict(input_array)
    return {"prediction": float(prediction[0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
    
