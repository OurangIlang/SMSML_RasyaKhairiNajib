
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path

BASE_DIR = Path(__file__).parent
mlflow.set_tracking_uri(f"file:{BASE_DIR}/mlruns")
mlflow.set_experiment("medical_insurance_regression")


def load_data(path="medical_insurance.csv"):
    df = pd.read_csv(path)
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

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", LinearRegression()),
        ]
    )
    return pipeline


def train():
    df = load_data()

    X = df.drop(columns=["charges", "log_charges"])
    y = df["log_charges"]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_pipeline(num_cols, cat_cols)

    mlflow.set_experiment("medical_insurance_regression")

    with mlflow.start_run(run_name="LinearRegression"):

        #train 
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # data origin scale
        y_pred_real = np.expm1(y_pred)
        y_test_real = np.expm1(y_test)
        rmse_real = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

        # Log params
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_param("log_transform", True)

        # Log metrics
        mlflow.log_metric("mae_log", mae)
        mlflow.log_metric("rmse_log", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("rmse_original", rmse_real)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=None
        )


        print("Training & logging completed.")
        print("RUN ID:", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    train()


