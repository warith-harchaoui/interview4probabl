from flask import Flask, request, jsonify
import joblib
import utils
import os
app = Flask(__name__)

# Load the model and hash (newest file)
model_path = f"best_pipeline_*.pkl"
model_path = max(glob(model_path), key=lambda f: -os.path.getctime(f))

hash = model_path.split("_")[-1].split(".")[0]


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
