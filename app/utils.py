"""
utils.py

Helper functions used by app/main.py: loading the trained model + preprocessing
pipeline, and turning a raw customer JSON payload into a churn prediction.
"""

import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
TRANSFORMER_PATH = os.path.join(BASE_DIR, "transformer.pkl")


def load_artifacts(model_path: str = MODEL_PATH, transformer_path: str = TRANSFORMER_PATH):
    """Load the trained model and fitted preprocessing pipeline from disk."""
    model = joblib.load(model_path)
    transformer = joblib.load(transformer_path)
    return model, transformer


def customer_dict_to_dataframe(customer: dict) -> pd.DataFrame:
    """Convert a single customer's raw JSON dict into a one-row DataFrame."""
    df = pd.DataFrame([customer])

    # Mirror the same cleanup applied during training so the transformer sees
    # data in the shape it was fit on.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df


def predict_churn(customer: dict, model, transformer) -> dict:
    """Run the full predict pipeline for one customer dict and return the result."""
    df = customer_dict_to_dataframe(customer)
    X_transformed = transformer.transform(df)

    probability = float(model.predict_proba(X_transformed)[0, 1])
    prediction = "Yes" if probability >= 0.5 else "No"

    return {
        "churn_probability": round(probability, 4),
        "churn_prediction": prediction,
    }
