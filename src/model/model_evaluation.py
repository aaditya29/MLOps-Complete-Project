import os
import json
import pickle
import mlflow
import logging
import dagshub
import numpy as np
import pandas as pd
import mlflow.sklearn
from src.logger import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Set your DagsHub MLflow tracking URI here
mlflow.set_tracking_uri(
    'https://dagshub.com/aaditya29/MLOps-Complete-Project.mlflow')
dagshub.init(repo_owner='aaditya29', repo_name="MLOps-Complete-Project",
             mlflow=True)  # Initialize DagsHub MLflow


def load_model(file_path: str):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error(
            'Unexpected error occurred while loading the model: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error(
            'Unexpected error occurred while loading the data: %s', e)
        raise
