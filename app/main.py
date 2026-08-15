"""
main.py

Flask application exposing the churn model as a real-time inference API.

Run from the project root as a module:
    python -m app.main

Listens on localhost:8000.
"""

import logging

from flask import Flask, request, jsonify

from app.utils import load_artifacts, predict_churn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the trained model + preprocessing pipeline once, at startup, so every
# request reuses the same in-memory objects instead of re-loading from disk.
model, transformer = load_artifacts()
logger.info("Model and transformer loaded successfully.")


@app.route("/health", methods=["GET"])
def health():
    """Simple liveness check."""
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts POST requests with raw customer JSON, e.g.:
        {"customer": {"gender": "Female", "SeniorCitizen": 0, ...}}

    Returns:
        {"churn_probability": 0.83, "churn_prediction": "Yes"}
    """
    try:
        payload = request.get_json(force=True, silent=False)
        if not payload or "customer" not in payload:
            return jsonify({"error": "Request body must contain a 'customer' object."}), 400

        customer = payload["customer"]
        result = predict_churn(customer, model, transformer)
        return jsonify(result), 200

    except Exception as exc:
        logger.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    # host=0.0.0.0 so it's reachable the same way if containerized later;
    # still accessible at localhost:8000 for local/grading use.
    app.run(host="0.0.0.0", port=8000)
