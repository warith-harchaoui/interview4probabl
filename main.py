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
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold


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

    ## Initial pipeline
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(stop_words='english', ngram_range=(1, 1)),  # Unigrams and bigrams
            ),
            (
                "decision_tree",
                DecisionTreeClassifier(
                    random_state=args.random_state,
                    max_depth=5,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features=None,
                    criterion="gini",
                ),
            ),
        ]
    )

    # Fit the initial pipeline
    pipeline.fit(train_data.data, y_train)

    # Evaluate and visualize the initial model's performance
    y_pred_val = pipeline.predict(val_data.data)
    proba_pred_val = pipeline.predict_proba(val_data.data)[:, 1]

    # Plot precision-recall and ROC curves for the initial model
    utils.plot_precision_recall_curve(y_val, proba_pred_val, "pr_curve_initial.png")
    utils.plot_roc_curve(y_val, proba_pred_val, "roc_curve_initial.png")

    # Additional Performance Metrics
    accuracy = accuracy_score(y_val, y_pred_val)
    f1 = f1_score(y_val, y_pred_val)
    class_report = classification_report(y_val, y_pred_val)
    conf_matrix = confusion_matrix(y_val, y_pred_val)

    utils.plot_confusion_matrix(conf_matrix, "initial_confusion_matrix.png")

    print(f"Initial Model Accuracy: {accuracy:.4f}")
    print(f"Initial Model F1 Score: {f1:.4f}")
    print("Initial Confusion Matrix:")
    print(conf_matrix)
    print("\nInitial Classification Report:")
    print(class_report)

    # Extract vocabulary and plot the initial decision tree
    vocabulary = pipeline.named_steps["tfidf"].get_feature_names_out()
    utils.plot_decision_tree(
        pipeline.named_steps["decision_tree"],
        vocabulary,
        ["Other", "Religion"],
        "initial_tree.png",  # Save initial tree visualization
    )

    ## Define the hyperparameter grid for tuning
    param_grid = {
        "tfidf__ngram_range": [(1, 2)],  # Unigrams, bigrams
        "decision_tree__max_depth": [20, 30, 40, 50],  # Different max depths
        "decision_tree__min_samples_split": [5, 10],    # Split thresholds
        "decision_tree__min_samples_leaf": [2, 4],      # Minimum samples in leaves
    }

    # Define the custom scorer for PR AUC
    pr_auc_scorer = make_scorer(pr_auc_score, needs_proba=True)

    # Set up Stratified 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)

    # Set up the GridSearchCV with PR AUC as the scoring metric
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=pr_auc_scorer,  # Use custom PR AUC scorer
        cv=skf,
        n_jobs=n_jobs,           # Parallelize across all available cores
        verbose=args.verbosity,  # Control verbosity level
    )

    # Fit the grid search
    grid_search.fit(train_data.data, y_train)

    # Display the best parameters and PR AUC score
    print(f"Best Params: {grid_search.best_params_}")
    print(f"Best PR AUC Score: {grid_search.best_score_}")

    # Evaluate the best model on validation set
    y_val_proba = grid_search.predict_proba(val_data.data)[:, 1]
    pr_auc_val = pr_auc_score(y_val, y_val_proba)
    print(f"Validation PR AUC: {pr_auc_val}")

    ## Save the best model and parameters
    best_params = grid_search.best_params_
    hash_string = hash(json.dumps(best_params, sort_keys=True))
    hash_string = str(hash_string)
    if hash_string[0] == "-":
        hash_string = "n" + hash_string[1:]
    joblib.dump(best_params, f'best_params_{hash_string}.pkl')
    best_pipeline = grid_search.best_estimator_
    joblib.dump(best_pipeline, f'best_pipeline_{hash_string}.pkl')

    ## Plot ROC and Precision-Recall Curves for the best model
    utils.plot_precision_recall_curve(y_val, y_val_proba, "pr_curve_best.png")
    utils.plot_roc_curve(y_val, y_val_proba, "roc_curve_best.png")

    ## Confusion Matrix for the validation set
    conf_matrix = confusion_matrix(y_val, grid_search.predict(val_data.data))
    utils.plot_confusion_matrix(conf_matrix, "best_confusion_matrix.png")

    ## Additional Analysis & Visualizations for the best model

    # Plot the Decision Tree from the best model
    vocabulary = best_pipeline.named_steps["tfidf"].get_feature_names_out()
    utils.plot_decision_tree(
        best_pipeline.named_steps["decision_tree"],
        vocabulary,
        ["Other", "Religion"],
        "best_tree.png",
    )

    # Feature Importance Analysis
    feature_importances = best_pipeline.named_steps["decision_tree"].feature_importances_
    relevant_vocabulary = np.array(vocabulary)[feature_importances > 0]
    feature_importances = feature_importances[feature_importances > 0]
    idx = np.argsort(feature_importances)[::-1]
    relevant_vocabulary = relevant_vocabulary[idx]
    feature_importances = feature_importances[idx]

    # Compute knee point for feature importance
    kneedle = KneeLocator(
        range(len(feature_importances)),
        feature_importances,
        curve="convex",
        direction="decreasing",
    )
    elbow_point_feature_importances = int(kneedle.elbow)
    print(
        f"Elbow point for feature importance: {elbow_point_feature_importances}"
    )
    relevant_vocabulary = relevant_vocabulary[: 3 * elbow_point_feature_importances]
    importances = feature_importances[: 3 * elbow_point_feature_importances]
    # plot feature importance
    utils.plot_feature_importance(relevant_vocabulary, importances, elbow_point_feature_importances, "feature_importances.png")


    ## Measure impact of specific words
    print("\nImpact of specific words (god, jesus, bible)")
    specific_words = ["god", "jesus", "bible"]
    train_positive_class_ratio = np.sum(y_train) / len(y_train)
    
    # Ensure the words exist in the vocabulary
    for word in specific_words:
        if word in vocabulary:
            idx = list(vocabulary).index(word)
            impact_score = utils.evaluate_feature_impact(
                best_pipeline, idx, train_positive_class_ratio
            )
            print(f"Impact of '{word}':\t{impact_score:.4f}")
        else:
            print(f"'{word}' is not in the vocabulary.")


    # Calculate Impact Scores for All Features
    impact_scores = []
    train_positive_class_ratio = np.sum(y_train) / len(y_train)
    for word in tqdm(relevant_vocabulary, desc="Computing impact scores"):
        idx = list(vocabulary).index(word)
        impact_score = utils.evaluate_feature_impact(
            best_pipeline, idx, train_positive_class_ratio
        )
        impact_scores.append((word, impact_score))

    # Sort by impact
    impact_scores = sorted(impact_scores, key=lambda x: x[1], reverse=True)

    # Extract the ordered vocabulary and impact scores
    ordered_vocabulary = [x[0] for x in impact_scores]
    ordered_impacts = [x[1] for x in impact_scores]

    # Find the elbow point for feature impact scores
    kneedle = KneeLocator(
        range(len(ordered_impacts)),
        ordered_impacts,
        curve="convex",
        direction="decreasing",
    )
    elbow_point = int(kneedle.elbow)

    # Filter words with impact greater than the threshold
    small_vocabulary = [x[0] for x in impact_scores[: 3 * elbow_point]]
    print(f"\nTop impactful words: {small_vocabulary[:10]}")

    # Save small vocabulary to JSON
    small_vocabulary_filename = os.path.join(args.data_home, "small_vocabulary.json")
    with open(small_vocabulary_filename, "wt", encoding="utf8") as file:
        json.dump(small_vocabulary, file, indent=2)

    print(f"Small vocabulary saved to {small_vocabulary_filename}")

    # Plot feature importance
    utils.plot_feature_importance(
        ordered_vocabulary,
        ordered_impacts,
        elbow_point,
        "feature_impact_scores.png",
    )
