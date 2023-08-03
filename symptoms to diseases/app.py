import pickle
from flask import Flask, request, jsonify

# Load the best model from the pickle file
with open("best_model.pickle", "rb") as f:
    best_model = pickle.load(f)

best_vectorizer = best_model["vectorizer"]
best_classifier = best_model["classifier"]

# Create the Flask application
app = Flask(__name__)

@app.route("/", methods=["POST"])
def predict_disease():
    try:
        # Get symptom descriptions from the request
        data = request.get_json()
        symptoms = data.get("symptoms")

        # Make predictions with the loaded model
        X_new = best_vectorizer.transform(symptoms)
        predictions = best_classifier.predict(X_new)

        # Prepare the response
        results = {"predictions": list(predictions)}

        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
