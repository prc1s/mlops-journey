import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import mlflow
from mlflow.models import infer_signature
import os
import dagshub
from sklearn.model_selection import train_test_split,GridSearchCV
from urllib.parse import urlparse 
from dotenv import load_dotenv

load_dotenv()
MLFLOW_TRACKING_URI      = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

def hyperparameter_tuning(x_train, y_train, param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=10, n_jobs=-1, verbose=2)
    grid_search.fit(x_train, y_train)
    return grid_search

#Load params from params.yaml
params = yaml.safe_load(open("params.yaml"))['train']

def train(data_path, model_path, random_state, n_estimators, max_depth):
    data = pd.read_csv(data_path) 
    x = data.drop(columns=["Outcome"])
    y = data['Outcome']
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Diabetes Prediction Model")
    with mlflow.start_run():
        #train test split
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
        signature = infer_signature(x_train,y_train)

        #Hyperparameters
        param_grid = {
            'n_estimators' : [90,100,110,120,130,140],
            'max_depth' : [5, 10, None],
            'min_samples_split' : [2,5],
            'min_samples_leaf' : [1,5]
        }
        
        #hyperparameter tuning
        grid_search = hyperparameter_tuning(x_train,y_train,param_grid)

        #get best model
        best_model = grid_search.best_estimator_

        #predict and evaluate the model
        y_pred = best_model.predict(x_test)
        accuracy = accuracy_score(y_pred, y_test)
        print(f"Accuracy Score: {accuracy}")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_params(grid_search.best_params_)

        #log confusion metrics and classification report
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        mlflow.log_text(str(cm), "confusion_matrics.txt")
        mlflow.log_text(str(cr), "classification_report.txt")

        import tempfile

        # 1) Save the trained model to a local temp directory (include the signature)
        tmpdir = tempfile.mkdtemp()
        mlflow.sklearn.save_model(sk_model=best_model, path=tmpdir, signature=signature)

        # 2) Log that entire directory as an artifact under the “model/” path
        mlflow.log_artifacts(tmpdir, artifact_path="model")

        print("✅ Model saved to MLflow artifacts under 'model/'")


        #create directory tp save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
 
        filname = model_path
        pickle.dump(best_model, open(filname, "wb"))
        print(f"Model saved to {model_path}")

if __name__=="__main__":
    train(params['data'], params['model'], params['random_state'], params['n_estimators'], params['max_depth'])



