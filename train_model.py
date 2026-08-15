"""
train_model.py

One-off training script (not part of the deployed API surface, but kept in the repo
for reproducibility / future retraining runs — see the Retraining section of README.md).

Trains a RandomForestClassifier on gold_churn_data.csv using the preprocessing
pipeline specified in the assignment instructions, and saves:
  - app/transformer.pkl  (fitted ColumnTransformer)
  - app/model.pkl         (fitted RandomForestClassifier)
"""

import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---- Load data ----
df = pd.read_csv("gold_churn_data.csv")

# Drop the index artifact column that pandas left behind when the CSV was exported
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# ---- Creating X and y (per assignment instructions) ----
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Step 1: Drop the 'customerID' column
X = X.drop(columns=["customerID"])

# Step 2: Convert 'TotalCharges' to numeric (handles spaces or non-numeric values)
X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

# Step 3: Convert target column 'y' to binary values
y = y.map({"Yes": 1, "No": 0})

# Step 4: Identify column types
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("Categorical columns:", categorical_cols)
print("Numerical columns:  ", numerical_cols)

# Step 5: Define preprocessing pipeline (no model yet)
# Note: newer scikit-learn (>=1.4) renamed OneHotEncoder's `sparse` kwarg to
# `sparse_output` — using the current name so this trains cleanly with the
# scikit-learn version pinned in requirements.txt.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="mean"), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ]
)

# Step 6: Apply the transformation to the data
X_cleaned = preprocessor.fit_transform(X)
print("Transformed feature matrix shape:", X_cleaned.shape)

# ---- Train/test split (kept consistent with earlier project modules) ----
X_train, X_test, y_train, y_test = train_test_split(
    X_cleaned, y, test_size=0.25, random_state=42, stratify=y
)

# ---- Train model ----
model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"Train accuracy: {train_acc:.4f}")
print(f"Test accuracy:  {test_acc:.4f}")

# ---- Save artifacts ----
joblib.dump(preprocessor, "app/transformer.pkl")
joblib.dump(model, "app/model.pkl")
print("Saved app/transformer.pkl and app/model.pkl")
