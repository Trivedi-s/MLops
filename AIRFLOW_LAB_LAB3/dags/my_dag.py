from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.model_development import load_data, data_preprocessing, build_model, evaluate_model
from src.success_email import send_success_email

default_args = {
    'owner': 'Shloka Trivedi',
    'start_date': datetime(2026, 2, 13),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'ML_Pipeline',
    default_args=default_args,
    description='ML Pipeline for Ad Click Prediction using Random Forest - By Shloka Trivedi',
    schedule='@daily',
    catchup=False,
    tags=['ml', 'advertising', 'random_forest', 'shloka']
)

load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag
)

data_preprocessing_task = PythonOperator(
    task_id='data_preprocessing_task',
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag
)

build_model_task = PythonOperator(
    task_id='build_model_task',
    python_callable=build_model,
    op_args=[data_preprocessing_task.output, "random_forest_model.sav"],
    dag=dag
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_model_task',
    python_callable=evaluate_model,
    op_args=[data_preprocessing_task.output, "random_forest_model.sav"],
    dag=dag
)

send_email_task = PythonOperator(
    task_id='send_email_task',
    python_callable=send_success_email,
    dag=dag
)

load_data_task >> data_preprocessing_task >> build_model_task >> evaluate_model_task >> send_email_task