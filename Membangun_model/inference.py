import mlflow.sklearn

MODEL_URI = "runs:/ddc3b3645efd4aa7992278501d0c6784/model"

model = mlflow.sklearn.load_model(MODEL_URI)

def predict(data):
    return model.predict(data)
