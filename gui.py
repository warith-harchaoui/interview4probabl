import streamlit as st
import joblib
from sklearn.pipeline import Pipeline
import numpy as np
import utils  # Import your utils module for the decision path and feature impact functions
from glob import glob
import os
import json


# Prediction function
def predict_and_interpret(text, model, small_vocabulary=None):
    # Predict the class of the input text
    prediction = model.predict([text])[0]
    probas = model.predict_proba([text])[0]

    # Interpretation: extract decision path
    decision_info = utils.extract_decision_path_for_sample(
        model, text, small_vocabulary=small_vocabulary, mute_analysis=True
    )

    return prediction, probas, decision_info


# Streamlit App
st.title("Topic Detection with Interpretation")
st.markdown("*Warith Harchaoui for Nicolas Delaforge, 7th and 8th October 2024*")

st.header("Introduction")
st.markdown(
    """
[Initial Instructions](https://github.com/warith-harchaoui/interview4probabl/blob/main/INSTRUCTIONS.md)

This project focuses on building a topic prediction model using NLP and machine learning techniques, specifically targeting the 20 Newsgroups dataset for binary classification. The goal is to classify whether a given email discusses religion. A decision tree classifier is trained using TF-IDF features, and its performance is assessed using key metrics like accuracy, precision, recall, F1 score, and ROC-AUC.

The project also emphasizes explainability by analyzing the features contributing to a given prediction. Visual tools such as precision-recall curves, ROC curves, feature importance plots, and decision trees help in understanding the classifier's performance and decision-making process. Further, strategies to improve the pipeline through hyperparameter tuning and feature engineering are explored, and the model is exposed via a web service using Streamlit and an attempt in Flask for real-time predictions with interpretability.
"""
)
#Markdown link
st.markdown("[🥸 Private GitHub Link](https://github.com/warith-harchaoui/interview4probabl)")


st.header("📈 Performance Analysis")

text = """
### Relevant Metrics for Assessing Performance

We can hesitate between ROC AUC and PR AUC.

We have a *detection problem* with two classes: `Religion` (13%) and `Other` (87%). The dataset is imbalanced, so **PR AUC** is the preferred good metric to use (ROC AUC is *too nice* with Chance).
"""
st.markdown(text)



st.image(
    "pr_curve_initial.png",
    caption="Starting Precision-Recall Curve",
    use_column_width=True,
)
st.image("roc_curve_initial.png", caption="Starting ROC Curve", use_column_width=True)

text = """

In fact, analyzing the confusion matrix in terms of business consequences should be done more thoroughly. For example, if the cost of a false positive (false alarm) is higher than the cost of a false negative (miss), we should focus on improving precision. If the cost of a false negative is higher, we should focus on improving recall.
"""
st.markdown(text)

st.image(
    "initial_confusion_matrix.png",
    caption="Starting Confusion Matrix",
    use_column_width=True,
)

text = """
These visualizations help us understand the trade-offs between precision and recall, as well as how well the model distinguishes between positive and negative classes.


### Analyzing the Classifier's Performance

To analyze the performance of our decision tree classifier:

- **Accuracy**: Measures the overall correctness of the model.
- **F1 Score**: Provides a balance between precision and recall.
- **Confusion Matrix**: Shows how often the classifier makes correct predictions versus mistakes (false positives and false negatives).
- **ROC and Precision-Recall Curves**: Visual representations of model performance across thresholds.

Using these metrics and visualizations (such as precision-recall and ROC curves), we can better understand where our model succeeds or fails.
"""
st.markdown(text)

st.header("🗣️ Explainability")

text = """
To make the predictions from the decision tree interpretable, we extract the decision path and calculate feature importance.

### Retrieving Features from the Decision Path

Using the function `utils.extract_decision_path_for_sample`, we can retrieve the list of words (features) that the decision tree used to make a prediction.

### Evaluating Feature Impact

For each node in the decision tree:
- The feature responsible for the split is identified.
- For each sample, we trace the path through the tree, calculating how the distribution of positive and negative samples changes at each split.
- This difference is used to calculate how impactful the feature was in making the prediction.

By evaluating these impacts, we can explain which words (features) contributed most to a specific prediction.

The work is done in the `utils.py` file at the `utils.evaluate_feature_impact` function.

With post processing, we can identity a knee in the curve of the feature impact scores. This knee (or a manually set multiple of this knee value) indicates the cut-off words for the most impactful features.
"""
st.markdown(text)

st.image(
    "feature_importances.png", caption="Classic sklearn Feature Importances", use_column_width=True
)

st.image(
    "feature_impact_scores.png",
    caption="Feature Impact Scores in function utils.evaluate_feature_impact",
    use_column_width=True,
)

st.header("➕ Improving the Pipeline")

text = """

1. **Hyperparameter Tuning**: Grid search or random search can optimize the performance of the decision tree. This includes tuning `max_depth`, `class_weight`, `criterion`, and `max_features`.
   
2. **Feature Engineering**:
   - **Expand Feature Space**: Consider adding linguistic or sentiment-based features in addition to TF-IDF.
   - **N-grams**: Experiment with larger N-grams (e.g., (1, 3)) to capture more context in the text.

3. **Handling Imbalanced Data**: Techniques like class weighting (e.g., using `class_weight="balanced"`) or resampling can help handle class imbalance.

4. **Other Classifiers**: Although a decision tree is simple and interpretable, more sophisticated models like Random Forest or Gradient Boosting could potentially yield better performance.

Cross-validation can ensure the model does not overfit and provides a better estimate of general performance.

  

"""
st.markdown(text)

st.image("initial_tree.png", caption="Initial Decision Tree", use_column_width=True)

st.image("best_tree.png", caption="Final Decision Tree", use_column_width=True)

st.header("🕸️ Setting up a Web Service")

st.markdown(
    "**I don't have much expertise on that. I used Flask for a simple API once. Here's my basic answer**"
)

text = """

### Flask-based API

```python
from flask import Flask, request, jsonify
import joblib
import utils

app = Flask(__name__)

# Load the model
model = joblib.load("path_to_best_model.joblib")

@app.route('/submit', methods=['POST'])
def submit_sample():
    data = request.json
    text = data.get('text')
    
    if text:
        # Predict and interpret
        prediction, probas, decision_info = utils.predict_and_interpret(text, model)
        response = {
            "prediction": prediction,
            "probabilities": probas.tolist(),
            "decision_path": decision_info['checked_words']
        }
        return jsonify(response), 200
    return jsonify({"error": "No text provided"}), 400

@app.route('/latest', methods=['GET'])
def get_latest_sample():
    # Retrieve the latest sample prediction and explanation from storage (e.g., database)
    pass

if __name__ == '__main__':
    app.run(debug=True)
```
    """

st.markdown(text)


st.header("👨‍💻 Small App")

# Load the model and hash (newest file)
model_path = f"best_pipeline_*.joblib"
model_path = max(glob(model_path), key=lambda f: -os.path.getctime(f))
model = joblib.load(model_path)

hash = model_path.split("_")[-1].split(".")[0]
small_vocabulary_filename = f"small_vocabulary_{hash}.json"
with open(small_vocabulary_filename, "rt", encoding="utf8") as file:
    small_vocabulary = json.load(file)

# Default text for the text area
default_text = """The role of religion in society has always been a significant topic of debate, with various perspectives 
on its influence on culture, morality, and governance."""

# User input
user_text = st.text_area("Enter text for classification:", value=default_text)

if st.button("Classify"):
    if len(user_text.strip()) > 0:
        prediction, probas, decision_info = predict_and_interpret(
            user_text, model, small_vocabulary
        )
        prediction, probas, decision_info = predict_and_interpret(user_text, model)

        # Display prediction
        st.write(f"**Prediction:** {'Religion' if prediction == 1 else 'Other'}")
        st.write(f"**Prediction Probabilities:**")
        s = round(probas[1] * 100)
        st.write(f"- Religion: {s}%")
        s = round(probas[0] * 100)
        st.write(f"- Other: {s}%")

        # Display decision path interpretation
        st.write("**Interpretation of Decision Path:**")
        ell = decision_info["checked_words"]
        ell = [w.replace("#--", "➖").replace("#++", "➕") for w in ell]
        st.write(f"Words checked in the decision path:")
        for e in ell:
            st.write(e)

    else:
        st.warning("Please enter some text for classification.")
