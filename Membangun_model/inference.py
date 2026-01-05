import mlflow
import mlflow.sklearn
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "mlruns" / "725492521526629918" / "models" / "m-583354fbd226428db6db0ce98f64119b" / "artifacts"

model = mlflow.sklearn.load_model(str(MODEL_PATH))

def predict(data_dict):
    df = pd.DataFrame([data_dict])
    return model.predict(df)
