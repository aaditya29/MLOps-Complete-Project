# for hyperparameter tuning of Logistic Regression model
import os
import re
import string
import mlflow
import dagshub
import warnings
import setuptools
import numpy as np
import scipy.sparse
import pandas as pd
import mlflow.sklearn

from nltk.corpus import stopwords
from xgboost import XGBClassifier
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")  # Ignore all warnings
warnings.simplefilter("ignore", UserWarning)  # Ignore UserWarnings
# Set pandas option to avoid silent downcasting
pd.set_option('future.no_silent_downcasting', True)


""" CONFIGURATION """
MLFLOW_TRACKING_URI = "https://dagshub.com/aaditya29/MLOps-Complete-Project.mlflow"
dagshub.init(repo_owner="aaditya29",
             repo_name="MLOps-Complete-Project", mlflow=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("LoR Hyperparameter Tuning")

"""Text Preprocessing Functions"""


def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()  # Initialize lemmatizer
    stop_words = set(stopwords.words("english"))  # Initialize stop words set

    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(f"[{re.escape(string.punctuation)}]",
                  " ", text)  # Remove punctuation
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split(
    ) if word not in stop_words])  # Lemmatization & stopwords removal

    return text.strip()  # Return preprocessed text


def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)

    # Apply text preprocessing
    df["review"] = df["review"].astype(str).apply(preprocess_text)

    # Filter for binary classification
    df = df[df["sentiment"].isin(["positive", "negative"])]
    df["sentiment"] = df["sentiment"].map({"negative": 0, "positive": 1})

    # Convert text data to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["review"])
    y = df["sentiment"]

    return train_test_split(X, y, test_size=0.2, random_state=42), vectorizer


""" Train and log model with hyperparameter tuning """


def train_and_log_model(X_train, X_test, y_train, y_test, vectorizer):
    """Trains a Logistic Regression model with GridSearch and logs results to MLflow."""

    param_grid = {
        "C": [0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"]
    }

    with mlflow.start_run():
        grid_search = GridSearchCV(
            LogisticRegression(), param_grid, cv=5, scoring="f1", n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Log all hyperparameter tuning runs
        for params, mean_score, std_score in zip(grid_search.cv_results_["params"],
                                                 grid_search.cv_results_[
                                                     "mean_test_score"],
                                                 grid_search.cv_results_["std_test_score"]):
            with mlflow.start_run(run_name=f"LR with params: {params}", nested=True):
                model = LogisticRegression(**params)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred),
                    "mean_cv_score": mean_score,
                    "std_cv_score": std_score
                }

                # Log parameters & metrics
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)

                print(
                    f"Params: {params} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")

        # Log the best model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_f1 = grid_search.best_score_

        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_score", best_f1)
        mlflow.sklearn.log_model(best_model, "model")

        print(f"\nBest Params: {best_params} | Best F1 Score: {best_f1:.4f}")


if __name__ == "__main__":
    (X_train, X_test, y_train, y_test), vectorizer = load_and_prepare_data(
        "/Users/adityamishra/Documents/MLOps-Capstone/notebooks/data.csv")
    train_and_log_model(X_train, X_test, y_train, y_test, vectorizer)
