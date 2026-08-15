# Customer-churn-api

A deployed customer churn prediction model: a real-time Flask inference API plus an
overnight batch-scoring pipeline that scores all active customers and logs basic
monitoring statistics.

## Project Structure

```
customer-churn-api/
│
├── app/
│   ├── __init__.py        # empty file, makes 'app' a module
│   ├── main.py             # Flask application, run via `python -m app.main`
│   ├── model.pkl            # Trained churn model (RandomForestClassifier)
│   ├── transformer.pkl      # Preprocessing pipeline (ColumnTransformer)
│   └── utils.py             # Helper functions (loading artifacts, prediction logic)
│
├── test_data/
│   ├── sample_input.json    # JSON input for /predict endpoint
│   └── all_customers.csv    # CSV input for batch scoring
│
├── batch.py                 # Batch scoring script
├── train_model.py           # One-off training script used to produce model.pkl / transformer.pkl
├── gold_churn_data.csv       # Source data train_model.py trains on
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── .gitignore
```

## Setup

```bash
pip install -r requirements.txt
```

The trained model (`app/model.pkl`) and preprocessing pipeline (`app/transformer.pkl`)
are already included in the repo, so no training step is required to run the API. If
you want to retrain from scratch (e.g. after updating `gold_churn_data.csv`), run:

```bash
python train_model.py
```

## Running the API

From the project root, start the server as a module:

```bash
python -m app.main
```

The Flask app listens on `localhost:8000` and exposes:

- `GET /health` — liveness check
- `POST /predict` — accepts `{"customer": {...}}` JSON, returns
  `{"churn_probability": 0.83, "churn_prediction": "Yes"}`

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @test_data/sample_input.json
```

## Batch Scoring

With the API running, score a full customer file overnight:

```bash
python batch.py --input test_data/all_customers.csv
```

This sends each row to `/predict` one at a time, writes `scored_customers.csv`
(original columns + `churn_probability`, `churn_prediction`, `scoring_status`), and
logs total requests, failures, and average churn probability to
`logs/batch_log.txt` (created automatically if it doesn't exist).

---

## Maintenance Plan

### 🧠 Retraining

Retrain on a **fixed monthly cadence**, or immediately if monitoring (below) flags a
meaningful drop in prediction quality. Retraining reruns `train_model.py` against the
latest export of `gold_churn_data.csv`, which refits both the `ColumnTransformer` and
the `RandomForestClassifier` together (never one without the other, since a stale
transformer silently corrupts inputs to a fresh model) and prints train/test accuracy
so a regression is visible immediately after each run.

Today that check is manual: whoever runs `train_model.py` reviews the printed
accuracy before copying the new `model.pkl` / `transformer.pkl` into `app/`. The
natural next step is to make this automatic — have the script compute
precision/recall/F1 on a held-out split (matching the Module 6 evaluation) and refuse
to overwrite the live artifacts unless the new model matches or beats the current
one on those metrics. That gate would stop a bad retrain from silently replacing a
working model, which the current script does not yet prevent on its own.

Any schema change upstream (new columns, renamed categories) should also trigger an
out-of-cycle retrain, since the preprocessing pipeline is fit to a specific column
set and will silently ignore unseen categories rather than error.

### 📉 Drift Detection

Track two kinds of drift using the batch job's own output as the data source:

1. **Prediction drift** — log the average churn probability from every batch run
   (already captured in `batch_log.txt`); a sustained shift compared to the trailing
   30-day average is an early warning sign worth investigating even before ground
   truth is available.
2. **Input drift** — periodically compare the distribution of key numeric inputs
   (`tenure`, `MonthlyCharges`, `TotalCharges`) and the frequency of categorical
   values (`Contract`, `PaymentMethod`) in incoming batches against the training
   distribution; large shifts suggest the customer base or product mix has changed
   enough that the model's assumptions may no longer hold.

Once actual churn outcomes become available for scored customers, compare them
against the logged predictions to recompute precision/recall on live data — this is
the ground-truth check that model-only drift metrics can't replace.

### 🏷️ Versioning

Store `model.pkl` and `transformer.pkl` together as a matched pair, tagged with a
version string embedded in the filename or a companion `model_version.txt` (e.g.
`v1.0.0-2026-08-15`) so a model is never deployed without its exact matching
preprocessing pipeline. Each retrain gets a new version tag and a short changelog
entry (date, data range used, headline metric changes) rather than overwriting
history in place, so any prediction can be traced back to the exact model version
and training data snapshot that produced it, and a bad release can be rolled back to
the previous tagged pair in minutes.
