#!/bin/bash

# Update pip
python -m pip install --upgrade pip

# Install base requirements
pip install -r requirements.txt

# Create virtual environments for Airflow and dbt
python -m venv airflow_venv
python -m venv dbt_venv

# Install Airflow dependencies in its virtual environment
source airflow_venv/bin/activate
pip install -r requirements-airflow.txt
deactivate

# Install dbt 
source dbt_venv/bin/activate
pip install -r requirements-dbt.txt
deactivate

echo "Dependencies installed successfully!"
echo "To use Airflow: source airflow_venv/bin/activate"
echo "To use dbt: source dbt_venv/bin/activate" 