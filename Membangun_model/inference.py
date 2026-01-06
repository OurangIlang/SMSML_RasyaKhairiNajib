import mlflow
import pandas as pd
import numpy as np

mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_URI = "models:/medical_insurance_model@production"
model = mlflow.pyfunc.load_model(MODEL_URI)

sample = pd.DataFrame([{
    "age": 35,
    "sex": "male",
    "bmi": 27.5,
    "children": 2,
    "smoker": "no",
    "region": "southeast"
}])

pred_log = model.predict(sample)
pred_real = np.expm1(pred_log)

print("Prediction:", float(pred_real[0]))
