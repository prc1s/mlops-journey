import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import yaml
import os
import mlflow
from mlflow.models import infer_signature
import pickle
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, GridSearchCV

load_dotenv()
MLFLOW_TRACKING_URI      = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

params = yaml.safe_load(open("params.yaml"))['train']
def evaluate(data_path, model_path):
    data = pd.read_csv(data_path)
    x = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model = pickle.load(open(model_path, 'rb'))

    predictions = model.predict(x)
    accuracy = accuracy_score(predictions, y)

    mlflow.log_metric("Accuracy", accuracy)

if __name__=="__main__":
    evaluate(params["data"], params["model"])