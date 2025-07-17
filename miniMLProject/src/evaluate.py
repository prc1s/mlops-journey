import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import yaml
import os
import mlflow
from mlflow.models import infer_signature
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV

os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/prc1s/miniMLProject.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "prc1s"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "2d8ebbdb54e757b2b945d893bf66803fae07a5aa"

params = yaml.safe_load(open("params.yaml"))['train']
def evaluate(data_path, model_path):
    data = pd.read_csv(data_path)
    x = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    model = pickle.load(open(model_path, 'rb'))

    predictions = model.predict(x)
    accuracy = accuracy_score(predictions, y)

    mlflow.log_metric("Accuracy", accuracy)

if __name__=="__main__":
    evaluate(params["data"], params["model"])