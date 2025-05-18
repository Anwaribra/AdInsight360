from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
from airflow.models import Variable
from airflow.hooks.base import BaseHook
from airflow.exceptions import AirflowException
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add scripts 
BASE_DIR = Path(__file__).parents[2].absolute()
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')
sys.path.append(str(SCRIPTS_DIR))

# Import task functions 
try:
    from extract_data import extract_reddit_data
    from load_to_snowflake import load_to_snowflake
except ImportError as e:
    logger.error(f"Failed to import task modules: {str(e)}")
    raise

DBT_PROJECT_DIR = os.getenv('DBT_PROJECT_DIR', str(BASE_DIR))
DBT_PROFILES_DIR = os.getenv('DBT_PROFILES_DIR', 'profiles')
MAX_ACTIVE_RUNS = 1

# DAG default 
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=30),
    'execution_timeout': timedelta(hours=1),
    'sla': timedelta(hours=2),
}

def task_failure_callback(context: Dict[str, Any]) -> None:
    """
    Enhanced error handling and logging for task failures
    
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
    """
    Wrapper function for extract_reddit_data with enhanced error handling.
    
    Args:
        context: Airflow context
        
    Returns:
        Result from extract_reddit_data
        
    Raises:
        AirflowException: If extraction fails
    """
    try:
        logger.info("Starting Reddit data extraction")
        result = extract_reddit_data()
        logger.info(f"Reddit data extraction completed successfully: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in extract_reddit_data: {str(e)}", exc_info=True)
        raise AirflowException(f"Reddit data extraction failed: {str(e)}")

def load_wrapper(**context):
    """
    Wrapper function for load_to_snowflake with enhanced error handling.
    
    Args:
        context: Airflow context
        
    Returns:
        Result from load_to_snowflake
        
    Raises:
        AirflowException: If loading fails
    """
    try:
        logger.info("Starting Snowflake data loading")
        result = load_to_snowflake()
        logger.info(f"Snowflake data loading completed successfully: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in load_to_snowflake: {str(e)}", exc_info=True)
        raise AirflowException(f"Snowflake data loading failed: {str(e)}")

# Define the DAG
dag = DAG(
    'reddit_marketing_pipeline',
    default_args=default_args,
    description='Pipeline to extract Reddit marketing data, load to Snowflake, and transform with dbt',
    schedule='0 */4 * * *',  # Run every 4 hours
    start_date=datetime(2025, 5, 17),  
    max_active_runs=MAX_ACTIVE_RUNS,
    tags=['reddit', 'marketing', 'snowflake', 'dbt'],
)










# Task 1: Extract Reddit data
extract_task = PythonOperator(
    task_id='extract_reddit_data',
    python_callable=extract_wrapper,
    on_failure_callback=task_failure_callback,
    doc_md="""
    Extracts marketing-related posts from Reddit using PRAW.
    
    This task:
    - Connects to Reddit using the PRAW library
    - Extracts posts from hot, new, and top categories
    - Prevents duplicate posts using a seen_ids set
    - Saves raw data to JSON files in the data/raw directory
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
    
    This task:
    - Creates tables if they don't exist
    - Handles data type conversions
    - Loads data using batch inserts for performance
    - Sets appropriate timestamps for data lineage
    """,
    dag=dag,
)

# Task 3: Run dbt staging models
dbt_staging = BashOperator(
    task_id='dbt_staging',
    bash_command=f'cd {DBT_PROJECT_DIR} && dbt run --select staging --profiles-dir {DBT_PROFILES_DIR}',
    on_failure_callback=task_failure_callback,
    doc_md="""
    Runs dbt staging models to transform raw data into standardized format.
    
    Models prepare data by:
    - Standardizing field names and types
    - Filtering invalid or irrelevant data
    - Creating initial transformation layer for analysis
    """,
    dag=dag,
)

# Task 4: Run dbt marts models
dbt_marts = BashOperator(
    task_id='dbt_marts',
    bash_command=f'cd {DBT_PROJECT_DIR} && dbt run --select marts --profiles-dir {DBT_PROFILES_DIR}',
    on_failure_callback=task_failure_callback,
    doc_md="""
    Runs dbt mart models to create business-level transformations.
    
    These models:
    - Aggregate data for analytical purposes
    - Build dimensional models for reporting
    - Create specialized views for business users
    - Calculate derived metrics for marketing insights
    """,
    dag=dag,
)

# Task 5: Run dbt tests
dbt_test = BashOperator(
    task_id='dbt_test',
    bash_command=f'cd {DBT_PROJECT_DIR} && dbt test --profiles-dir {DBT_PROFILES_DIR}',
    on_failure_callback=task_failure_callback,
    doc_md="""
    Runs data quality tests to ensure data integrity and business rules.
    
    Tests verify:
    - Referential integrity
    - Business rule compliance
    - Data completeness
    - Valid ranges and formats
    """,
    dag=dag,
)

# Task 6: Generate dbt documentation (optional task)
dbt_docs = BashOperator(
    task_id='dbt_docs',
    bash_command=f'cd {DBT_PROJECT_DIR} && dbt docs generate --profiles-dir {DBT_PROFILES_DIR}',
    on_failure_callback=task_failure_callback,
    doc_md="""
    Generates dbt documentation for the project.
    
    The documentation includes:
    - Data lineage diagrams
    - Table and column descriptions
    - Test coverage information
    - Model relationships
    """,
    dag=dag,
)

# Set task dependencies
extract_task >> load_task >> dbt_staging >> dbt_marts >> dbt_test >> dbt_docs