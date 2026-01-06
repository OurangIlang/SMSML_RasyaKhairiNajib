import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("medical_insurance_regression")

mlflow.sklearn.autolog(
    log_models=True,
    registered_model_name="medical_insurance_model"
)

def load_data():
    df = pd.read_csv("medical_insurance.csv")
    df = df.drop_duplicates()
    df["log_charges"] = np.log1p(df["charges"])
    return df

def build_pipeline(num_cols, cat_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", LinearRegression()),
        ]
    )

def train():
    df = load_data()

    X = df.drop(columns=["charges", "log_charges"])
    y = df["log_charges"]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_pipeline(num_cols, cat_cols)

    with mlflow.start_run(run_name="LinearRegression"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mlflow.log_metric("rmse_log", rmse)


        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="medical_insurance_model"
    )
        
if __name__ == "__main__":
    train()
