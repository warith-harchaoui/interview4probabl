"""
Text Classification using Decision Trees

This script builds a text classification pipeline using a Decision Tree classifier and the 20newsgroups dataset. 
The pipeline includes TF-IDF for feature extraction and MaxAbsScaler for scaling the features. 
It performs cross-validation, evaluates the performance using Precision-Recall and ROC curves, 
and extracts the decision path for explainability.

Author: Warith Harchaoui
"""

import os
from typing import Union, Dict, List, Tuple
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, precision_recall_curve, roc_curve, auc
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
from collections import OrderedDict
import yaml

from sklearn.tree import plot_tree

class LabelBinarizer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for binarizing labels for classification tasks.

    Args:
        binary_target (list): A list of target classes to binarize.

    Attributes:
        binary_target (list): Stores the target classes to be binarized.
    """

    def __init__(self, binary_target: List[str]):
        self.binary_target = binary_target

    def fit(self, X, y=None):
        """Fit method, does nothing since this is a transformer."""
        return self

    def transform(self, X) -> np.ndarray:
        """
        Binarizes the labels based on the provided target classes.

        Args:
            X: The dataset to transform.

        Returns:
            np.ndarray: Binarized labels.
        """
        y_binarized = np.zeros(len(X.target), dtype=bool)
        for binary_target_ in self.binary_target:
            binary_target_idx = X.target_names.index(binary_target_)
            y_binarized |= X.target == binary_target_idx
        return y_binarized


def cleaner_text(input_text: str) -> str:
    """
    Cleans input text by replacing special characters and removing extra spaces.

    Args:
        input_text (str): The input text to clean.

    Returns:
        str: The cleaned text.
    """
    text = str(input_text).replace("_", " ").replace("--", " ")
    lines = [" ".join(line.strip().split()) for line in text.split("\n")]
    return "\n".join(lines)


def load_newsgroup_data(
    binary_target: List[str], data_home: str, random_state: int
) -> Tuple:
    """
    Load and preprocess the 20newsgroups dataset, applying binarization.

    Args:
        binary_target (list): The target classes for binarization.
        data_home (str): Path to the data directory.
        random_state (int): Seed for reproducibility.

    Returns:
        Tuple: (train_data, val_data, y_train, y_val)
    """
    news_group_folder = os.path.join(data_home, "20newsgroups")
    train_data = fetch_20newsgroups(
        data_home=news_group_folder,
        random_state=random_state,
        subset="train",
        remove=("headers", "footers", "quotes"),
    )
    val_data = fetch_20newsgroups(
        data_home=news_group_folder,
        random_state=random_state,
        subset="test",
        remove=("headers", "footers", "quotes"),
    )

    # Clean the dataset
    train_data.data = [cleaner_text(text) for text in train_data.data]
    val_data.data = [cleaner_text(text) for text in val_data.data]

    # Binarize the labels
    y_train = LabelBinarizer(binary_target).fit_transform(train_data)
    y_val = LabelBinarizer(binary_target).fit_transform(val_data)

    return train_data, val_data, y_train, y_val


def build_pipeline(
    lang: str = None,
    random_state: int = 123,
    max_depth: int = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str = None,
    criterion: str = "gini",
    class_weight=None,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Pipeline:
    """
    Constructs a classification pipeline with a TF-IDF vectorizer, a scaler, and a Decision Tree classifier.

    Args:
        lang (str): Language for stop words.
        random_state (int): Random seed for the Decision Tree.
        max_depth (int): Maximum depth of the Decision Tree.
        min_samples_split (int): Minimum samples required to split an internal node.
        min_samples_leaf (int): Minimum samples required to be at a leaf node.
        max_features (str): The number of features to consider when looking for the best split.
        criterion (str): The function to measure the quality of a split.
        class_weight (str): Weights associated with classes in the form {class_label: weight}.
        ngram_range (Tuple[int, int]): The range of N-grams to consider.

    Returns:
        Pipeline: A sklearn pipeline object.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(stop_words=lang, ngram_range=ngram_range),
            ),  # N-grams (unigrams and bigrams)
            ("scaler", MaxAbsScaler()),
            (
                "decision_tree",
                DecisionTreeClassifier(
                    random_state=random_state,
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


def plot_precision_recall_curve(
    y_true: np.ndarray, y_proba: np.ndarray, output_image_path: str = "pr_curve.pdf"
) -> None:
    """
    Plots the Precision-Recall curve and saves the plot.

    Args:
        y_true (np.ndarray): True binary labels.
        y_proba (np.ndarray): Predicted probabilities.
        output_image_path (str): Path to save the plot.
    """
    precision_chance = sum(y_true) / len(y_true)

    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    precision_dummy_0, recall_dummy_0, _ = precision_recall_curve(
        y_true, [0.0] * len(y_true)
    )
    precision_dummy_1, recall_dummy_1, _ = precision_recall_curve(
        y_true, [1.0] * len(y_true)
    )

    plt.clf()
    plt.figure(figsize=(5, 5))
    plt.plot(
        recall,
        precision,
        label="Decision Tree",
        linewidth=2,
        color="#007AFF")
    plt.plot(
        [0, 1],
        [precision_chance] * 2,
        label="Chance",
        color="#DA8FFF",
        linewidth=1,
    )
    plt.plot(
        recall_dummy_0,
        precision_dummy_0,
        label="Constant 0",
        linewidth=1,
        color="#FFD60A",
        linestyle="dashed",
    )
    plt.plot(
        recall_dummy_1,
        precision_dummy_1,
        label="Constant 1",
        linewidth=2,
        color="#FFB340",
        linestyle="dotted",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    # Remove lines around the plot
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", which="both", bottom=False, top=False)
    ax.tick_params(axis="y", which="both", left=False, right=False)
    auc_value = auc(recall, precision)
    plt.title(f"Precision-Recall Curve (AUC = {round(auc_value * 100)}%)")
    plt.savefig(output_image_path)


def plot_roc_curve(
    y_true: np.ndarray, y_proba: np.ndarray, output_image_path: str = "roc_curve.pdf"
) -> None:
    """
    Plots the ROC curve and saves the plot.

    Args:
        y_true (np.ndarray): True binary labels.
        y_proba (np.ndarray): Predicted probabilities.
        output_image_path (str): Path to save the plot.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fpr_dummy_0, tpr_dummy_0, _ = roc_curve(y_true, [0.0] * len(y_true))
    fpr_dummy_1, tpr_dummy_1, _ = roc_curve(y_true, [1.0] * len(y_true))

    plt.clf()
    plt.figure(figsize=(5, 5))
    plt.plot(
        fpr,
        tpr,
        label="Decision Tree",
        linewidth=2,
        color="#007AFF")
    plt.plot(
        [0, 1],
        [0, 1],
        label="Chance",
        color="#DA8FFF",
        linewidth=1
    )
    plt.plot(
        fpr_dummy_0,
        tpr_dummy_0,
        label="Constant 0",
        linewidth=1,
        c="#FFD60A",
        linestyle="dashed",
    )
    plt.plot(
        fpr_dummy_1,
        tpr_dummy_1,
        label="Constant 1",
        linewidth=2,
        c="#FFB340",
        linestyle="dotted",
    )
    auc_value = auc(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title(f"ROC Curve (AUC = {round(auc_value * 100)}%)")
    # Remove lines around the plot
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", which="both", bottom=False, top=False)
    ax.tick_params(axis="y", which="both", left=False, right=False)
    plt.savefig(output_image_path)


def pr_auc_scorer(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Custom scorer to compute the area under the Precision-Recall curve.

    Args:
        y_true (np.ndarray): True binary labels.
        y_proba (np.ndarray): Predicted probabilities.

    Returns:
        float: Area under the Precision-Recall curve (AUC).
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return auc(recall, precision)


def render_in_string(d: Union[Dict, List, Tuple, str, float, int]) -> str:
    """
    Converts various data types (dict, list, tuple, str, float, int) to a well-formatted string representation.
    
    For dictionaries, lists, and tuples, returns a YAML string representation.
    For simple types (str, float, int), returns them as strings directly.

    Args:
        d (Union[Dict, List, Tuple, str, float, int]): The input data.

    Returns:
        str: The string or YAML formatted string representation of the input.
    """
    if isinstance(d, OrderedDict):
        # Convert ordered dictionaries to regular dictionaries
        return render_in_string(dict(d))
    elif isinstance(d, (dict, list, tuple)):
        # Convert tuples to lists for consistent YAML formatting
        if isinstance(d, tuple):
            d = list(d)
        # Convert dictionaries, lists, and tuples (as lists) to YAML
        return yaml.dump(d, sort_keys=True, default_flow_style=False, indent=2)
    else:
        # Return strings and other simple types as is
        return str(d)




def cross_validate_pipeline(
    train_data: List[str],
    y_train: np.ndarray,
    param_grid: Dict,
    random_state: int,
    n_jobs: int = -1,
    verbose: bool = False,
) -> Dict:
    """
    Performs cross-validation using a GridSearch over a specified parameter grid.

    Args:
        train_data (List[str]): Training data.
        y_train (np.ndarray): Training labels.
        param_grid (Dict): Grid of parameters for the pipeline.
        random_state (int): Seed for reproducibility.
        n_jobs (int): Number of jobs for parallel processing.
        verbose (bool): Verbosity level for the cross-validation process.

    Returns:
        Dict: Best parameters from GridSearch.
    """
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),  # N-gram range in pipeline
            ("scaler", MaxAbsScaler()),
            ("decision_tree", DecisionTreeClassifier(random_state=random_state)),
        ]
    )

    # Custom scorer using PR AUC
    pr_auc = make_scorer(pr_auc_scorer, needs_proba=True)

    # Cross-validation with GridSearch with stratification on the target labels
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=stratified_kfold,
        scoring=pr_auc,
        verbose=verbose,
        n_jobs=n_jobs,
    )

    grid_search.fit(train_data, y_train)
    s = render_in_string(grid_search.best_params_)
    print(f"Best parameters found:\n\t{s}")
    print(f"Best PR AUC: {grid_search.best_score_:.4f}")

    return grid_search.best_params_


def evaluate_feature_impact(
    pipeline: Pipeline,
    feature_idx: int,
    train_positive_class_ratio: float,
    positive_class: int = 1,
) -> float:
    """
    Evaluates the impact of a specific feature on positive class prediction in a Decision Tree.

    Args:
        pipeline (Pipeline): Trained pipeline with a Decision Tree.
        feature_idx (int): The index of the feature to evaluate.
        train_positive_class_ratio (float): The ratio of positive samples in the training data.
        positive_class (int): The class considered as positive.

    Returns:
        float: The impact of the feature on predicting the positive class.
    """
    tree = pipeline.named_steps["decision_tree"].tree_
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    values = tree.value

    total_samples = values[0][0].sum()
    feature_impact = 0.0

    # Traverse all nodes in the decision tree
    for node_id in range(n_nodes):
        if feature[node_id] == feature_idx:
            samples_in_node = values[node_id][0].sum()
            positive_samples_in_node = values[node_id][0, positive_class]
            fraction_positive_before = (
                positive_samples_in_node / samples_in_node if samples_in_node > 0 else 0
            )
            impact_before = fraction_positive_before - train_positive_class_ratio

            left_child = children_left[node_id]
            right_child = children_right[node_id]
            samples_left = values[left_child][0].sum()
            samples_right = values[right_child][0].sum()
            positive_samples_left = values[left_child][0, positive_class]
            positive_samples_right = values[right_child][0, positive_class]
            fraction_positive_after = (
                (
                    samples_left * (positive_samples_left / samples_left)
                    + samples_right * (positive_samples_right / samples_right)
                )
                / (samples_left + samples_right)
                if samples_left + samples_right > 0
                else 0
            )
            impact_after = fraction_positive_after - train_positive_class_ratio
            impact_change = abs(impact_after - impact_before)
            feature_impact += impact_change * (samples_in_node / total_samples)

    return feature_impact


def extract_decision_path_for_sample(
    pipeline: Pipeline,
    text: str,
    small_vocabulary: list = None,
    mute_analysis: bool = True,
) -> Dict:
    """
    Extracts the decision path and rules for predicting a single sample using a Decision Tree.

    Args:
        pipeline (Pipeline): The trained pipeline with a Decision Tree.
        text (str): A sample text (article) to analyze.
        small_vocabulary (list): A list of words to check in the decision path.
        mute_analysis (bool): Whether to mute detailed analysis output.

    Returns:
        Dict: Contains sample text, checked words, predicted class, and predicted probability.
    """
    checked_words = []
    clf_name = "decision_tree"
    tree = pipeline.named_steps[clf_name].tree_
    features = tree.feature
    tfidf = pipeline.named_steps["tfidf"]
    threshold = tree.threshold
    vocabulary = tfidf.vocabulary_
    vocabulary = sorted(vocabulary.keys(), key=lambda x: vocabulary[x])

    sample_pred = pipeline.predict([text])[0]
    sample_proba = pipeline.predict_proba([text])[0]
    samples_processed_until_clf = [text]

    # Apply transformations up to the decision tree
    for step in pipeline.named_steps:
        if step == clf_name:
            break
        samples_processed_until_clf = pipeline.named_steps[step].transform(
            samples_processed_until_clf
        )

    node_indicator = pipeline.named_steps[clf_name].decision_path(
        samples_processed_until_clf
    )
    leaf_id = pipeline.named_steps[clf_name].apply(samples_processed_until_clf)
    node_index = node_indicator.indices[
        node_indicator.indptr[0] : node_indicator.indptr[1]
    ]

    if not mute_analysis:
        print(f"Rules used to predict sample:\n")

    for node_id in node_index:
        if leaf_id[0] == node_id:
            continue

        # Check if feature is below or above the threshold
        if samples_processed_until_clf[0, features[node_id]] <= threshold[node_id]:
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        node_word = vocabulary[features[node_id]]
        if not (small_vocabulary is None) and not (node_word in small_vocabulary):
            continue
        b = threshold_sign == ">"
        s = "#++ " if b else "#-- "
        checked_words.append(s + node_word)
        if not mute_analysis:
            y = "Yes" if s == "+" else "No"
            print(f"Decision: Is word '{node_word}' present in the text? {y}")

    if not mute_analysis:
        print(f"Sample:\n\t{text.strip()}")
        print(f"Checked words: {', '.join(checked_words)}")

    d = OrderedDict()
    d["sample"] = text.strip()
    d["checked_words"] = checked_words
    return d

def plot_confusion_matrix(conf_matrix, output_image_path):
    plt.clf()
    plt.figure(figsize=(5, 5))
    plt.matshow(conf_matrix, cmap="Blues", fignum=1)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            color = "white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black"
            plt.text(x=j, y=i, s=conf_matrix[i, j], va="center", ha="center", color=color)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_image_path)

def plot_decision_tree(clf, vocabulary, class_names, output_image_path):
    plt.clf()
    plt.figure(figsize=(120, 80))
    plot_tree(clf, filled=True, feature_names=vocabulary, class_names=class_names)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=False, top=False)
    ax.tick_params(axis='y', which='both', left=False, right=False)
    plt.savefig(output_image_path)

def plot_feature_importance(relevant_vocabulary, importances, elbow_point, output_image_path):
    plt.clf()
    plt.figure(figsize=(10, 5))
    plt.bar(relevant_vocabulary, importances, color="#007AFF")
    plt.xticks(rotation=45)
    plt.axvline(x=elbow_point, color="red", linestyle="--", label="Elbow Point")
    plt.xlabel("Words")
    plt.ylabel("Feature Importance")
    plt.title("Decision Tree Feature Importances")
    plt.legend()
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=False, top=False)
    ax.tick_params(axis='y', which='both', left=False, right=False)
    plt.tight_layout()
    plt.savefig(output_image_path)

def plot_feature_impact(ordered_vocabulary, ordered_impacts, elbow_point, output_image_path):
    plt.clf()
    plt.figure(figsize=(10, 5))
    plt.bar(ordered_vocabulary[:3*elbow_point], ordered_impacts[:3*elbow_point], label="Impact Score", color="#007AFF")
    plt.xticks(rotation=45)
    plt.axvline(x=elbow_point, color="red", linestyle="--", label="Elbow Point")
    plt.xlabel("Words")
    plt.ylabel("Impact Score")
    plt.legend()
    plt.title("Feature Impact Scores")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=False, top=False)
    ax.tick_params(axis='y', which='both', left=False, right=False)
    plt.tight_layout()
    plt.savefig(output_image_path)
