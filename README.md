# Topic Prediction with NLP and Decision Trees

This project is a case study focused on building a topic prediction model using NLP and machine learning techniques. The primary goal is to classify emails from the 20 Newsgroups dataset into a binary classification task: determining whether an email discusses religion. 

We use a decision tree classifier trained on TF-IDF features, and we analyze its performance through various metrics such as precision, recall, F1 score, and ROC-AUC. The project also incorporates explainability features, enabling us to interpret the model’s predictions. Finally, we expose the model through a web service using Streamlit and Flask, allowing real-time predictions with decision path explanations.

---

## Features

- Binary classification for detecting religion-related emails.
- Performance analysis using precision-recall curves, ROC curves, and confusion matrices.
- Explainability features for interpreting which words influenced the decision tree's predictions.
- Web interface for real-time classification using Streamlit.
- RESTful API using Flask for submitting and retrieving predictions.

---

## Installation

To set up the project, follow these steps:

### 1. Clone the repository

```bash
git clone https://github.com/warith-harchaoui/interview4probabl
cd interview4probabl
```

Create a python environment or go to my webpage if you don't know how

[🥸 Tech tips](https://harchaoui.org/warith/4ml/#install)

```bash
pip install -r requirements.txt
```

or manually
```bash
pip install numpy scikit-learn matplotlib joblib streamlit flask tqdm kneed flask
```

# Usage

## Running the Main Pipeline
You can execute the main pipeline using the `main.py` script. This will train the model, evaluate its performance, generate plots, and analyze decision paths.

```bash
python main.py --random_state 123 --data_home ./data --n_jobs -1 --verbosity 1
```

The generated performance visualizations will be saved as `.png` files, and you will also see metrics like accuracy, F1 score, and confusion matrix in the terminal.

```bash
python main.py -h
usage: main.py [-h] [--random_state RANDOM_STATE] [--data_home DATA_HOME] [--n_jobs N_JOBS]
               [--verbosity VERBOSITY]

Text Classification using Decision Trees

options:
  -h, --help            show this help message and exit
  --random_state RANDOM_STATE
                        Random seed for reproducibility
  --data_home DATA_HOME
                        Path to data folder
  --n_jobs N_JOBS       Number of jobs for parallel processing. -1 uses all available cores.
  --verbosity VERBOSITY
                        Verbosity level for GridSearchCV. 0 = quiet.
```

## Running the Web Application

### Streamlit (UI)
To run the Streamlit app that provides a web interface for classifying text and visualizing the decision path, use the following command:

```bash
streamlit run gui.py
```
This will open the Streamlit interface in your browser, where you can input text and receive predictions.

### Flask (API, I am bad at it)
To set up a RESTful API for submitting text samples and receiving predictions, you can run the `Flask` app as follows

```bash
python app.py
```
You can then send POST requests to the /submit endpoint with a sample and retrieve the prediction, as well as the decision path used to classify it.
