from airflow import DAG 
from airflow.operators.python import PythonOperator
from datetime import datetime

##Define Task
def preprocess_data():
    print("Preprocessing data...")

def training_model():
    print("training model...")

def evaluate_model():
    print("Evaluating model...")


with DAG(
    "ml_pipeline",
    start_date=datetime(2024,1,1),
    schedule='@weekly'
) as dag:
    #define task
    preprocess = PythonOperator(task_id="preprocess_data", python_callable=preprocess_data)
    train = PythonOperator(task_id="training_model", python_callable=training_model)
    evaluate = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)

    preprocess >> train >> evaluate