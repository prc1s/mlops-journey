from airflow import DAG 
from airflow.operators.python import PythonOperator
from datetime import datetime

def start_number(**context):
    context["ti"].xcom_push(key='current_value', value=10)
    print("Starting value 10")

def add_five(**context):
    current_value = context["ti"].xcom_pull(key='current_value', task_ids='start_task')
    new_value = current_value+5
    context["ti"].xcom_push(key='current_value', value=new_value)
    print(f"five has been added to {current_value} so the result is {new_value}")

def multiply_two(**context):
    current_value = context["ti"].xcom_pull(key='current_value', task_ids='add_five_task')
    new_value = current_value*2
    context["ti"].xcom_push(key='current_value', value=new_value)
    print(f"{current_value} has been multiplied by two so the result is {new_value}")

def subtract_three(**context):
    current_value = context["ti"].xcom_pull(key='current_value', task_ids='multiply_two_task')
    new_value = current_value-3
    context["ti"].xcom_push(key='current_value', value=new_value)
    print(f"{current_value} has been subtracted by three so the result is {new_value}")

def square(**context):
    current_value = context["ti"].xcom_pull(key='current_value', task_ids='subtract_three_task')
    new_value = current_value**2
    context["ti"].xcom_push(key='current_value', value=new_value)
    print(f"{current_value} has been squared so the result is {new_value}")

with DAG(
    dag_id='math_sequence_operation',
    start_date=datetime(2024,1,1),
    schedule='@once',
    catchup=False
) as dag:
    
    start_task=PythonOperator(
        task_id = 'start_task',
        python_callable=start_number,
    )

    add_five_task=PythonOperator(
        task_id = 'add_five_task',
        python_callable=add_five,
    )

    multiply_two_task=PythonOperator(
        task_id = 'multiply_two_task',
        python_callable=multiply_two,
    )

    subtract_three_task=PythonOperator(
        task_id = 'subtract_three_task',
        python_callable=subtract_three,
    )

    square_three_task=PythonOperator(
        task_id = 'square_three_task',
        python_callable=square,
    )


    start_task >> add_five_task >> multiply_two_task >> subtract_three_task >> square_three_task



