from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import sys
import os

# Add the scripts directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../scripts'))

# Import the functions from your scripts
from extract_data import extract_reddit_data
from load_to_snowflake import load_to_snowflake

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'reddit_marketing_pipeline',
    default_args=default_args,
    description='Pipeline to extract Reddit marketing data and load to Snowflake',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['reddit', 'marketing', 'snowflake', 'dbt'],
)

# Task 1: Extract Reddit data
extract_task = PythonOperator(
    task_id='extract_reddit_data',
    python_callable=extract_reddit_data,
    dag=dag,
)

# Task 2: Load data to Snowflake
load_task = PythonOperator(
    task_id='load_to_snowflake',
    python_callable=load_to_snowflake,
    dag=dag,
)

# Task 3: Run dbt staging models
dbt_staging = BashOperator(
    task_id='dbt_staging',
    bash_command='cd /home/anwar/AdInsight360 && dbt run --select staging',
    dag=dag,
)

# Task 4: Run dbt marts models
dbt_marts = BashOperator(
    task_id='dbt_marts',
    bash_command='cd /home/anwar/AdInsight360 && dbt run --select marts',
    dag=dag,
)

# Task 5: Run dbt tests
dbt_test = BashOperator(
    task_id='dbt_test',
    bash_command='cd /home/anwar/AdInsight360 && dbt test',
    dag=dag,
)

# Set task dependencies
extract_task >> load_task >> dbt_staging >> dbt_marts >> dbt_test
