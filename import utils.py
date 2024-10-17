import utils
import json
import os
import argparse
from kneed import KneeLocator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    log_loss,
    precision_recall_curve,
    auc,
    make_scorer,
)
from tqdm import tqdm
import joblib
import sys

from sklearn.datasets import fetch_20newsgroups
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin


def pr_auc_score(y_true, y_proba):
    """Custom scorer function to calculate the PR AUC"""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return auc(recall, precision)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Text Classification using Decision Trees"
    )
    parser.add_argument(
        "--random_state", type=int, default=123, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--data_home", type=str, default="./", help="Path to data folder"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of jobs for parallel processing. -1 uses all available cores.",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        help="Verbosity level for GridSearchCV. 0 = quiet.",
    )
    args = parser.parse_args()

    verbose = args.verbosity > 0
    n_jobs = args.n_jobs

    # Define the binary target classes for classification
    binary_target = ["talk.religion.misc", "alt.atheism", "soc.religion.christian"]

    # Load and preprocess the dataset
    train_data, val_data, y_train, y_val = utils.load_newsgroup_data(
        binary_target, args.data_home, args.random_state
    )
    print(
        f"Train Data Size:\t{len(train_data.data)}\nVal Data Size:\t{len(val_data.data)}"
    )

    print(
        f"Positive ratio in training set:\t{round(100*np.sum(y_train) / len(y_train))}%"
    )

    max_depth = 20
    min_samples_split = 2
    min_samples_leaf = 1
    max_features = None
    criterion = "gini"
    class_weight = None
    lang = "english"
    ngram_range = (1, 2)

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(stop_words=lang, ngram_range=ngram_range),
            ),
            (
                "decision_tree",
                DecisionTreeClassifier(
                    random_state=args.random_state,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    criterion=criterion,
                    class_weight=class_weight,
                ),
            ),
        ]
    )

    # Define hyperparameters for grid search
    param_grid = {
        'decision_tree__max_depth': [10, 20, 30],
        'decision_tree__min_samples_split': [2, 5, 10],
        'decision_tree__min_samples_leaf': [1, 2, 4],
    }

    # Define the custom scorer for PR AUC
    pr_auc_scorer = make_scorer(pr_auc_score, needs_proba=True)

    # Set up Stratified 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)

    # GridSearchCV with PR AUC as the scoring metric
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=pr_auc_scorer,
        cv=skf,
        n_jobs=n_jobs,
        verbose=args.verbosity,
    )

    # Fit the grid search
    grid_search.fit(train_data.data, y_train)

    print(f"Best Params: {grid_search.best_params_}")
    print(f"Best PR AUC Score: {grid_search.best_score_}")

    # Evaluate on validation set
    y_val_proba = grid_search.predict_proba(val_data.data)[:, 1]
    pr_auc_val = pr_auc_score(y_val, y_val_proba)
    print(f"Validation PR AUC: {pr_auc_val}")
