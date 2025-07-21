import os
import warnings
import sys

import mlflow.sklearn
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll import scope


import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rsme=np.sqrt(mean_squared_error(actual, pred))
    mae=mean_absolute_error(actual, pred)
    r2=r2_score(actual, pred)
    return rsme, mae, r2


def train_model(params, train_x, train_y, test_x, test_y):
    model = ElasticNet(alpha=params["alpha"], l1_ratio=params["l1_ratio"], random_state=42, selection="cyclic")
    load_dotenv()
    MLFLOW_TRACKING_URI      = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Wine Quality --ALL MODELS")
    with mlflow.start_run(nested=True):

        model.fit(train_x, train_y,)
        pred = model.predict(test_x)
        rsme, mae, r2 = eval_metrics(pred, test_y)

        mlflow.log_param("alpha", params["alpha"])
        mlflow.log_param("l1_ratio", params["l1_ratio"])
        mlflow.log_param("random_state", 42)
        mlflow.log_param("selection", "cyclic")
        mlflow.log_metric("rsme", rsme)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        singnature = infer_signature(train_x, train_y)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store!="file":
            mlflow.sklearn.log_model(
                model,"model",registered_model_name="ElasticnetWineModel", signature=singnature
            )
        else:
            mlflow.sklearn.log_model(model, "model", signature=singnature)
        
        return {"loss" : rsme, "rsme" : rsme, "mae" : mae,  "r2" : r2, "status" : STATUS_OK, "model" : model }



if __name__ == "__main__":

    ##Data Ingestion
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )

    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception("Failed to download the data.")



    #train test split

    train, test = train_test_split(data, test_size=0.3)
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train["quality"]
    test_y = test["quality"]


    def objective(params):
        result = train_model(
            params,
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y
        )
        return result

    # alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    # l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    

    #AWS Remote Server Setup
    load_dotenv()
    MLFLOW_TRACKING_URI      = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Wine Quality --BEST MODELS")
    with mlflow.start_run():
        space = {
        "alpha":    hp.loguniform("alpha",   np.log(1e-6), np.log(1e-3)),
        "l1_ratio": hp.uniform("l1_ratio",    0.0, 1.0)
        }
        
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=10,
            trials=trials
        )

        best_model = sorted(trials.results,key=lambda x: x["loss"])[0]

        mlflow.log_params(best)

        mlflow.log_metric("rsme", best_model["rsme"])
        mlflow.log_metric("mae", best_model["mae"])
        mlflow.log_metric("r2", best_model["r2"])

        singnature = infer_signature(train_x, train_y)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store!="file":
            mlflow.sklearn.log_model(
                best_model["model"],"model",registered_model_name="ElasticnetWineModel", signature=singnature
            )
        else:
            mlflow.sklearn.log_model(best_model["model"], "model", signature=singnature)
