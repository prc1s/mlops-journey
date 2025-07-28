from airflow import DAG 
from airflow.operators.python import PythonOperator
from datetime import datetime
from airflow.decorators import task

with DAG(
    dag_id = 'math_sequence_flow_with_taskflow',
    start_date = datetime(2024,1,1),
    catchup = False,
) as dag:
    
    @task
    def start_number():
        initial_value = 10
        print(f"Starting value = {initial_value}")
        return initial_value
    
    @task
    def add_five(n):
        added_five = n + 5 
        print(f"added five to {n} the result is {add_five}")
        return added_five
    
    @task 
    def multiply_two(n):
        multiplied_two = n * 2 
        print(f"{n} multiplied by 2 is {multiplied_two}")
        return multiplied_two

    @task 
    def subtract_three(n):
        subtracted_three = n - 3
        print(f"{n} subtract by 3 is {subtracted_three}")
        return subtracted_three

    @task
    def square(n):
        squared = n ** 2
        print(f"{n} squared is {squared}")
        return squared
    
    start_value = start_number()
    added_five = add_five(start_value)
    multiplied_two = multiply_two(added_five)
    subtracted_three = subtract_three(multiplied_two)
    squared = square(subtracted_three)