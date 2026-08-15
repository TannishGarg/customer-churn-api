"""
batch.py

Overnight batch scoring pipeline. Reads a CSV of active customers, sends each
row as JSON to the running /predict API one customer at a time, collects the
results into scored_customers.csv, and logs request/failure/probability
statistics to logs/batch_log.txt.

Usage:
    python batch.py --input test_data/all_customers.csv
"""

import argparse
import logging
import os
import time

import pandas as pd
import requests

API_URL = "http://localhost:8000/predict"

# Columns the /predict endpoint expects in the "customer" object — i.e. every
# original customer attribute except the identifier and the label, which the
# model was never trained to see at inference time.
DROP_COLUMNS = ["Unnamed: 0", "customerID", "Churn"]


def setup_logging(log_dir: str = "logs", log_file: str = "batch_log.txt") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger("batch")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers if re-run in the same process

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(console_handler)

    return logger


def row_to_customer_payload(row: pd.Series) -> dict:
    """Build the {"customer": {...}} payload the API expects from one CSV row."""
    customer = row.drop(labels=[c for c in DROP_COLUMNS if c in row.index]).to_dict()
    return {"customer": customer}


def score_customers(input_path: str, output_path: str, logger: logging.Logger, timeout: int = 10):
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} customers from {input_path}")

    total_requests = 0
    failures = 0
    probabilities = []

    predictions = []
    probabilities_col = []
    statuses = []

    for idx, row in df.iterrows():
        total_requests += 1
        customer_id = row.get("customerID", f"row_{idx}")
        payload = row_to_customer_payload(row)

        try:
            response = requests.post(API_URL, json=payload, timeout=timeout)
            if response.status_code == 200:
                result = response.json()
                prob = result.get("churn_probability")
                pred = result.get("churn_prediction")
                probabilities.append(prob)
                probabilities_col.append(prob)
                predictions.append(pred)
                statuses.append("success")
            else:
                failures += 1
                logger.error(
                    f"Customer {customer_id}: non-200 response ({response.status_code}) - {response.text}"
                )
                probabilities_col.append(None)
                predictions.append(None)
                statuses.append(f"failed:http_{response.status_code}")

        except requests.exceptions.RequestException as exc:
            failures += 1
            logger.error(f"Customer {customer_id}: request exception - {exc}")
            probabilities_col.append(None)
            predictions.append(None)
            statuses.append("failed:exception")

    df["churn_probability"] = probabilities_col
    df["churn_prediction"] = predictions
    df["scoring_status"] = statuses
    df.to_csv(output_path, index=False)

    avg_probability = sum(probabilities) / len(probabilities) if probabilities else float("nan")

    logger.info(f"Total requests: {total_requests}")
    logger.info(f"Failures: {failures}")
    logger.info(f"Successful predictions: {total_requests - failures}")
    logger.info(f"Average churn probability (successful only): {avg_probability:.4f}" if probabilities else "Average churn probability: N/A (no successful predictions)")
    logger.info(f"Scored results written to {output_path}")

    return {
        "total_requests": total_requests,
        "failures": failures,
        "average_probability": avg_probability,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch-score customers against the churn /predict API.")
    parser.add_argument("--input", required=True, help="Path to CSV of customers to score.")
    parser.add_argument("--output", default="scored_customers.csv", help="Path to write scored results.")
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("=== Starting batch scoring run ===")
    start = time.time()

    try:
        stats = score_customers(args.input, args.output, logger)
    except Exception as exc:
        logger.exception(f"Batch run failed: {exc}")
        raise

    elapsed = time.time() - start
    logger.info(f"=== Batch scoring run complete in {elapsed:.2f}s === {stats}")


if __name__ == "__main__":
    main()
