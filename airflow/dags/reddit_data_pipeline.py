from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
from airflow.exceptions import AirflowException
from datetime import datetime, timedelta
import logging
import sys
import os
from typing import Dict, Any


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), '../../scripts')
sys.path.append(SCRIPTS_DIR)


from extract_data import extract_reddit_data
from load_to_snowflake import load_to_snowflake

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': True,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=30),
    'execution_timeout': timedelta(hours=1),
    'sla': timedelta(hours=2),
}

def task_failure_callback(context: Dict[str, Any]) -> None:
    """
    Enhanced error handling and logging for task failures.
    
    Args:
        context: Airflow context containing task instance and error information
    """
    task_instance = context['task_instance']
    exception = context.get('exception')
    
    error_msg = (
        f"Task Failed - DAG: {task_instance.dag_id}, "
        f"Task: {task_instance.task_id}, "
        f"Execution Time: {task_instance.execution_date}, "
        f"Exception: {str(exception)}"
    )
    
    logger.error(error_msg)

def extract_wrapper(**context):
    """Wrapper function for extract_reddit_data with better error handling."""
    try:
        return extract_reddit_data()
    except Exception as e:
        logger.error(f"Error in extract_reddit_data: {str(e)}")
        raise AirflowException(f"Reddit data extraction failed: {str(e)}")

def load_wrapper(**context):
    """Wrapper function for load_to_snowflake with better error handling."""
    try:
        return load_to_snowflake()
    except Exception as e:
        logger.error(f"Error in load_to_snowflake: {str(e)}")
        raise AirflowException(f"Snowflake data loading failed: {str(e)}")

# Define the DAG
dag = DAG(
    'reddit_marketing_pipeline',
    default_args=default_args,
    description='Pipeline to extract Reddit marketing data, load to Snowflake, and transform with dbt',
    schedule_interval='0 */4 * * *',  # Run every 4 hours
    start_date=days_ago(1),
    catchup=False,
    tags=['reddit', 'marketing', 'snowflake', 'dbt'],
    doc_md="""
    # Reddit Marketing Data Pipeline
    
    This DAG orchestrates the following tasks:
    1. Extracts marketing data from Reddit
    2. Loads the data into Snowflake
    3. Transforms the data using dbt models
    4. Runs data quality tests
    
    ## Dependencies
    - Reddit API credentials
    - Snowflake connection
    - dbt project setup
    """,
)

DBT_PROJECT_DIR = '/home/anwar/AdInsight360'

# Task 1: Extract Reddit data
extract_task = PythonOperator(
    task_id='extract_reddit_data',
    python_callable=extract_wrapper,
    on_failure_callback=task_failure_callback,
    doc_md="""
    Extracts marketing-related posts from Reddit using PRAW.
    Saves raw data to JSON files in the data/raw directory.
    """,
    dag=dag,
)

# Task 2: Load data to Snowflake
load_task = PythonOperator(
    task_id='load_to_snowflake',
    python_callable=load_wrapper,
    on_failure_callback=task_failure_callback,
    doc_md="""
    Loads extracted Reddit data into Snowflake staging tables.
    Handles data type conversions and maintains data integrity.
    """,
    dag=dag,
)

# Task 3: Run dbt staging models
dbt_staging = BashOperator(
    task_id='dbt_staging',
    bash_command=f'cd {DBT_PROJECT_DIR} && dbt run --select staging --profiles-dir profiles',
    on_failure_callback=task_failure_callback,
    doc_md="""
    Runs dbt staging models to transform raw data into standardized format.
    """,
    dag=dag,
)

# Task 4: Run dbt marts models
dbt_marts = BashOperator(
    task_id='dbt_marts',
    bash_command=f'cd {DBT_PROJECT_DIR} && dbt run --select marts --profiles-dir profiles',
    on_failure_callback=task_failure_callback,
    doc_md="""
    Runs dbt mart models to create business-level transformations.
    """,
    dag=dag,
)

# Task 5: Run dbt tests
dbt_test = BashOperator(
    task_id='dbt_test',
    bash_command=f'cd {DBT_PROJECT_DIR} && dbt test --profiles-dir profiles',
    on_failure_callback=task_failure_callback,
    doc_md="""
    Runs data quality tests to ensure data integrity and business rules.
    """,
    dag=dag,
)

# Set task dependencies
extract_task >> load_task >> dbt_staging >> dbt_marts >> dbt_test 
