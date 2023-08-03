import pickle
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin

# Load the best model from the pickle file
with open("best_model.pickle", "rb") as f:
    best_model = pickle.load(f)

best_vectorizer = best_model["vectorizer"]
best_classifier = best_model["classifier"]

# Create the Flask application
app = Flask(__name__)

cors = CORS(app, resources={
            r'/*': {"origins": ["http://localhost:3000", "*"], "supports_credentials": True}})


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route("/", methods=["POST"])
def predict_disease():
    try:
        data = request.get_json()
        symptoms = data.get("symptoms")

        # Make predictions with the loaded model
        X_new = best_vectorizer.transform(symptoms)
        class_labels = best_classifier.classes_  # Get the class labels

        # Check if the classifier supports probability estimates
        if hasattr(best_classifier, "predict_proba"):
            probabilities = best_classifier.predict_proba(
                X_new)  # Get the probability scores
        elif hasattr(best_classifier, "decision_function"):
            decision_values = best_classifier.decision_function(X_new)
            probabilities = (decision_values - decision_values.min(axis=1)[:, None]) / (
                decision_values.max(axis=1)[:, None] - decision_values.min(axis=1)[:, None])
        else:
            raise ValueError(
                "The selected classifier does not support probability estimates.")

        # Prepare the response
        results = []
        for i, prob in enumerate(probabilities):
            percentage_match = max(prob) * 100
            disease_prediction = {
                "disease": class_labels[i],
                "percentage_match": round(percentage_match, 2)
            }
            results.append(disease_prediction)

        return jsonify(results), 200
    except Exception as e:
        print('cehck', e)
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
