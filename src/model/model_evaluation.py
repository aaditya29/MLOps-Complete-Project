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


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logging.info('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logging.error('Error during model evaluation: %s', e)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the metrics: %s', e)
        raise


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.debug('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise
