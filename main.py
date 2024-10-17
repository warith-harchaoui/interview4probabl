import utils
import json
import os
import argparse
from kneed import KneeLocator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, log_loss
from tqdm import tqdm
import joblib 
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Classification using Decision Trees")
    parser.add_argument("--random_state", type=int, default=123, help="Random seed for reproducibility")
    parser.add_argument("--data_home", type=str, default="./", help="Path to data folder")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of jobs for parallel processing. -1 uses all available cores.")
    parser.add_argument("--verbosity", type=int, default=1, help="Verbosity level for GridSearchCV. 0 = quiet.")
    args = parser.parse_args()

    verbose = args.verbosity > 0
    n_jobs = args.n_jobs

    # Define the binary target classes for classification
    binary_target = ["talk.religion.misc", "alt.atheism", "soc.religion.christian"]

    # Load and preprocess the dataset
    train_data, val_data, y_train, y_val = utils.load_newsgroup_data(binary_target, args.data_home, args.random_state)
    print(f"Train Data Size:\t{len(train_data.data)}\nVal Data Size:\t{len(val_data.data)}")

    print(f"Positive ratio in training set:\t{round(100*np.sum(y_train) / len(y_train))}%")

    # Build pipeline with n-gram range (1, 2) for unigrams and bigrams
    pipeline = utils.build_pipeline(
        lang="english", 
        random_state=args.random_state, 
        max_depth=20, # Customize max_depth for better tree interpretability
        min_samples_split=2, 
        min_samples_leaf=1, 
        max_features=None, 
        criterion="gini"
    )
    
    pipeline.fit(train_data.data, y_train)

    # Evaluate and visualize performance
    y_pred_val = pipeline.predict(val_data.data)
    proba_pred_val = pipeline.predict_proba(val_data.data)[:, 1]

    # Plot precision-recall and ROC curves
    utils.plot_precision_recall_curve(y_val, proba_pred_val, "pr_curve_initial.png")
    utils.plot_roc_curve(y_val, proba_pred_val, "roc_curve_initial.png")

    # Additional Performance Metrics
    accuracy = accuracy_score(y_val, y_pred_val)
    f1 = f1_score(y_val, y_pred_val)
    class_report = classification_report(y_val, y_pred_val)

    conf_matrix = confusion_matrix(y_val, y_pred_val)

    utils.plot_confusion_matrix(conf_matrix, "initial_confusion_matrix.png")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

    vocabulary = pipeline.named_steps["tfidf"].get_feature_names_out()
    vocabulary = [w for w in vocabulary]

    # Plot decision tree
    
    utils.plot_decision_tree(pipeline.named_steps["decision_tree"], vocabulary, ["Other", "Religion"], "initial_tree.png")
    
    ## Explainability

    # Classic feature importance
    feature_importances = pipeline.named_steps["decision_tree"].feature_importances_
    relevant_vocabulary = np.array(vocabulary)[feature_importances>0]
    feature_importances = feature_importances[feature_importances>0]
    idx = np.argsort(feature_importances)[::-1]
    relevant_vocabulary = relevant_vocabulary[idx]
    feature_importances = feature_importances[idx]

    # COmpute knee point
    kneedle = KneeLocator(range(len(feature_importances)), feature_importances, curve="convex", direction="decreasing")
    elbow_point_feature_importances = int(kneedle.elbow)
    print(f"Elbow point for classic Decision Tree feature importance: {elbow_point_feature_importances}")
    relevant_vocabulary = relevant_vocabulary[:3*elbow_point_feature_importances]
    importances = feature_importances[:3*elbow_point_feature_importances]

    
    utils.plot_feature_importance(relevant_vocabulary, importances, elbow_point_feature_importances, "feature_importances.png")



    # Example of decision path extraction for a sample
    sample_idx = 0
    sample = val_data.data[sample_idx]
    decision_info = utils.extract_decision_path_for_sample(pipeline, sample, mute_analysis=True)
    s = utils.render_in_string(decision_info)
    print(f"Text analysis for sample {sample_idx}:\n{s}")

    # Measure the impact of some words
    print("\nImpact of some words")
    some_words_to_measure = ["god", "jesus", "bible"]
    train_positive_class_ratio = np.sum(y_train) / len(y_train)
    for word in some_words_to_measure:
        idx = vocabulary.index(word)
        impact_score = utils.evaluate_feature_impact(
            pipeline, idx, train_positive_class_ratio
        )
        print(f"Impact of '{word}':\t{impact_score:.4f}")

    # Sort the vocabulary by impact
    impact_scores = []
    for word in tqdm(relevant_vocabulary, desc = "Computing impact scores"):
        idx = vocabulary.index(word)
        impact_score = utils.evaluate_feature_impact(
            pipeline, idx, train_positive_class_ratio
        )
        impact_scores.append((word, impact_score))

    impact_scores = sorted(impact_scores, key=lambda x: x[1], reverse=True)
    print("\nTop 10 most impactful words:")
    for word, impact_score in impact_scores[:10]:
        print(f"{word}:\t{impact_score:.4f}")

    ordered_vocabulary = [x[0] for x in impact_scores]
    ordered_impacts = [x[1] for x in impact_scores]

    # Find the knee point
    kneedle = KneeLocator(
        range(len(ordered_impacts)),
        ordered_impacts,
        curve="convex",
        direction="decreasing",
    )
    elbow_point = int(kneedle.elbow)
    threshold = ordered_impacts[elbow_point]
    filtered_words = [
        (word, impact) for word, impact in impact_scores if impact >= threshold
    ]
    print(f"\nTop {elbow_point} most impactful words:")
    for e, (word, impact) in enumerate(impact_scores[:3 * elbow_point]):
        print(f"{word}:\t{impact:.4f}")
        if e == elbow_point:
            print("-" * 10)

    
    utils.plot_feature_impact(ordered_vocabulary, ordered_impacts, elbow_point, "feature_impact_scores.png")

    vocabulary_filename = f"vocabulary.json"
    with open(vocabulary_filename, "wt", encoding="utf8") as file:
        json.dump(ordered_vocabulary, file, indent=2)


    small_vocabulary = ordered_vocabulary[:5*elbow_point]
    
    top = 2
    # Take positive samples
    print(f"\nTop {top} positive samples:")
    pos_samples = [(sample, label) for sample, label in zip(val_data.data, y_val) if label == 1]
    neg_samples = [(sample, label) for sample, label in zip(val_data.data, y_val) if label == 0]
    for samples in [pos_samples, neg_samples]:
        for sample, label in samples[:top]:
            print(f"Text:\n{sample}\n\nLabel: {label}\n")
            decision_info = utils.extract_decision_path_for_sample(pipeline, sample, small_vocabulary=small_vocabulary, mute_analysis=True)
            s = utils.render_in_string(decision_info)
            print(f"Path of words in the tree:\n{s}")
            print("-"*10)
            print("")

    ## Improving our pipeline

    # Cross validation to find the best hyperparameters
    basic = 1.0 / train_positive_class_ratio
    print(f"Basic weight to outweight the positive class ratio: {basic:.4f}")
    weights = [{0: 1, 1: float(basic * r)} for r in np.arange(2, 5)]

    depths = list(range(5, 5*8+1, 5))

    # Define the hyperparameter grid for tuning
    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)], 
        "tfidf__stop_words": [None, "english"],  
        "decision_tree__class_weight": [None, "balanced"] + weights, 
        "decision_tree__max_depth": depths,  
        "decision_tree__min_samples_split": [2, 5, 10], 
        "decision_tree__min_samples_leaf": [2, 5], 
        "decision_tree__criterion": ["gini", "entropy"], 
        "decision_tree__max_features": ['sqrt', 'log2'],  
    }
    best_params_hash = hash(json.dumps(param_grid, sort_keys=True))

    small_vocabulary_filename = f"small_vocabulary_{best_params_hash}.json"
    with open(small_vocabulary_filename, "wt", encoding="utf8") as file:
        json.dump(small_vocabulary, file, indent=2)

    best_params_filename = f"best_params_{best_params_hash}.json"
    model_filename = f"best_pipeline_{best_params_hash}.joblib"
    if os.path.exists(best_params_filename):
        with open(best_params_filename, "rt", encoding="utf8") as file:
            best_params = json.load(file)
    else:
        best_params = utils.cross_validate_pipeline(
            train_data.data, y_train, param_grid, args.random_state, n_jobs=n_jobs, verbose=verbose
        )
        with open(best_params_filename, "wt", encoding="utf8") as file:
            json.dump(best_params, file, indent=2, sort_keys=True)

    s = utils.render_in_string(best_params)
    print(f"Best parameters:\n{s}")

    if os.path.exists(model_filename):
        best_pipeline = joblib.load(model_filename)
    else:
        # Rebuild and evaluate the best pipeline with optimal parameters
        best_pipeline = utils.build_pipeline(
            lang=best_params["tfidf__stop_words"],
            random_state=args.random_state,
            max_depth=best_params["decision_tree__max_depth"],
            min_samples_split=best_params["decision_tree__min_samples_split"],
            min_samples_leaf=best_params["decision_tree__min_samples_leaf"],
            max_features=best_params["decision_tree__max_features"],
            criterion=best_params["decision_tree__criterion"],
            class_weight=best_params["decision_tree__class_weight"],
            ngram_range=best_params["tfidf__ngram_range"],
        )
        best_pipeline.fit(train_data.data, y_train)
        # Save the best pipeline with joblib
        joblib.dump(best_pipeline, model_filename)

    # plot tree for best pipeline
    
    utils.plot_decision_tree(best_pipeline.named_steps["decision_tree"], vocabulary, ["Other", "Religion"], "best_tree.png")

    y_pred_val = best_pipeline.predict(val_data.data)
    proba_pred_val = best_pipeline.predict_proba(val_data.data)[:, 1]

    # Plot curves for the optimized model
    utils.plot_precision_recall_curve(y_val, proba_pred_val, "pr_curve_best.png")
    utils.plot_roc_curve(y_val, proba_pred_val, "roc_curve_best.png")

    conf_matrix = confusion_matrix(y_val, y_pred_val)
    utils.plot_confusion_matrix(conf_matrix, "best_confusion_matrix.png")

