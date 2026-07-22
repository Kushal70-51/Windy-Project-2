"""
daily_feedback.py

Run this manually at the end of each day, once your plant's actual
SCADA/meter export for that day is available:

    python daily_feedback.py <path_to_actual_meter_csv>

It does TWO things:
    1. Adds an "Actual Generation (MW)" column to features_log.csv
       (the case store) for every matching timestamp -- this is what
       lets similarity_retrieval.py eventually show the LLM "in similar
       situations, actual generation was X" instead of just predictions.
    2. Computes and prints error metrics (MAE, RMSE, MAPE, Bias) comparing
       predicted vs actual generation, and appends a row to a running
       accuracy log CSV so you can track accuracy over time.

No LLM, no ML training -- pure deterministic comparison and file update.

Expected actual-meter CSV format: at least a timestamp column and a
power column, matching TIMESTAMP_COLUMN / POWER_COLUMN_MW below (adjust
to your SCADA export's real column names). Negative power readings are
treated as 0 (no generation), consistent with earlier data-cleaning
decisions in this project.
"""

import csv
import sys
import datetime
from pathlib import Path

import config

# ---- Adjust these to match your actual SCADA export's column names ----
TIMESTAMP_COLUMN = "TimeStamp"
POWER_COLUMN_MW = "Active Power (MW)"   # expects MW already -- convert first if your export is in kW
# -------------------------------------------------------------------

ACTUAL_COLUMN_NAME = "Actual Generation (MW)"
ACCURACY_LOG_PATH = Path("accuracy_reports") / f"{config.PLANT_NAME}_daily_accuracy.csv"


def _load_actual_readings(actual_csv_path: str) -> dict:
    """
    Reads the actual meter CSV and returns {time_label: actual_mw}, with
    time_label formatted as "%Y-%m-%d %H:%M" to match features_log.csv's
    Time column, and negative readings clipped to 0.
    """
    readings = {}
    with open(actual_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if TIMESTAMP_COLUMN not in reader.fieldnames or POWER_COLUMN_MW not in reader.fieldnames:
            raise SystemExit(
                f"Expected columns '{TIMESTAMP_COLUMN}' and '{POWER_COLUMN_MW}' not found. "
                f"Available columns: {reader.fieldnames}\n"
                f"Update TIMESTAMP_COLUMN / POWER_COLUMN_MW at the top of this script to match."
            )
        for row in reader:
            raw_ts = row[TIMESTAMP_COLUMN].strip()
            try:
                # Accept both "YYYY-MM-DD HH:MM" and "YYYY-MM-DD HH:MM:SS"
                dt = datetime.datetime.strptime(raw_ts[:16], "%Y-%m-%d %H:%M")
            except ValueError:
                continue
            time_label = dt.strftime("%Y-%m-%d %H:%M")
            try:
                mw = max(0.0, float(row[POWER_COLUMN_MW]))
            except (TypeError, ValueError):
                continue
            readings[time_label] = mw
    return readings


def _update_case_store_with_actuals(actual_readings: dict) -> tuple:
    """
    Reads features_log.csv, fills in ACTUAL_COLUMN_NAME for every row
    whose Time matches an actual reading, and rewrites the file using the
    same schema-safe (dict-keyed, DictWriter) approach as
    prediction_store.py -- so adding this new column can never shift or
    corrupt existing values, even for rows written before this column
    existed.

    Returns (updated_count, matched_rows) where matched_rows is a list of
    (predicted_mw, actual_mw) pairs for the error-metric calculation below.
    """
    csv_path = config.FEATURES_LOG_DIR / f"{config.PLANT_NAME}_features_log.csv"
    if not csv_path.exists():
        raise SystemExit(f"Case store not found at {csv_path} -- run the main pipeline first.")

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        original_fieldnames = list(reader.fieldnames)
        rows = list(reader)

    fieldnames = list(original_fieldnames)
    if ACTUAL_COLUMN_NAME not in fieldnames:
        fieldnames.append(ACTUAL_COLUMN_NAME)

    updated_count = 0
    matched_rows = []
    for row in rows:
        time_label = row.get("Time")
        if time_label in actual_readings:
            actual_mw = actual_readings[time_label]
            row[ACTUAL_COLUMN_NAME] = str(actual_mw)
            updated_count += 1

            predicted_raw = row.get("Predicted Generation (MW)")
            try:
                predicted_mw = float(predicted_raw)
                matched_rows.append((predicted_mw, actual_mw))
            except (TypeError, ValueError):
                pass

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return updated_count, matched_rows


def _compute_error_metrics(matched_rows: list) -> dict:
    """Computes MAE, RMSE, MAPE (skipping zero-actual rows to avoid
    divide-by-zero), and Bias (mean signed error, +ve = over-predicting)."""
    if not matched_rows:
        return {}

    n = len(matched_rows)
    abs_errors = [abs(pred - actual) for pred, actual in matched_rows]
    signed_errors = [pred - actual for pred, actual in matched_rows]
    squared_errors = [(pred - actual) ** 2 for pred, actual in matched_rows]

    mae = sum(abs_errors) / n
    rmse = (sum(squared_errors) / n) ** 0.5
    bias = sum(signed_errors) / n

    pct_errors = [
        abs(pred - actual) / actual for pred, actual in matched_rows if actual > 0
    ]
    mape = (sum(pct_errors) / len(pct_errors) * 100) if pct_errors else None

    return {
        "n_matched_blocks": n,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape_pct": round(mape, 2) if mape is not None else None,
        "bias": round(bias, 4),
    }


def _log_accuracy(metrics: dict) -> None:
    """Appends today's metrics as a new row to the running accuracy log,
    so accuracy over time can be reviewed later."""
    ACCURACY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    row = {"date": datetime.date.today().isoformat(), **metrics}
    file_exists = ACCURACY_LOG_PATH.exists()

    with open(ACCURACY_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def sync_historic_case_actuals() -> int:
    """Join every compatible SCADA CSV in ``historic_cases`` to the case store.

    This is safe to call before every forecast: it only fills actual values
    for timestamps already captured by the pipeline and does not create
    duplicate rows or accuracy-log entries.
    """
    case_store = config.FEATURES_LOG_DIR / f"{config.PLANT_NAME}_features_log.csv"
    if not case_store.exists():
        return 0

    readings = {}
    for csv_path in config.HISTORIC_CASES_DIR.glob("*.csv"):
        try:
            readings.update(_load_actual_readings(str(csv_path)))
        except (OSError, SystemExit):
            # A folder can contain auxiliary CSVs; only files matching the
            # configured SCADA timestamp/power columns are relevant here.
            continue
    if not readings:
        return 0
    updated_count, _ = _update_case_store_with_actuals(readings)
    return updated_count


def run_daily_feedback(actual_csv_path: str) -> None:
    print(f"Loading actual meter readings from: {actual_csv_path}")
    actual_readings = _load_actual_readings(actual_csv_path)
    print(f"  Found {len(actual_readings)} usable actual readings.")

    print("\nUpdating case store with actual generation values...")
    updated_count, matched_rows = _update_case_store_with_actuals(actual_readings)
    print(f"  Updated {updated_count} rows in the case store with actual generation.")

    if not matched_rows:
        print("\n[INFO] No predicted+actual pairs matched by timestamp -- "
              "no error metrics to compute. Check that your pipeline was "
              "running (and logging predictions) at these times.")
        return

    print("\nComputing error metrics...")
    metrics = _compute_error_metrics(matched_rows)
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    _log_accuracy(metrics)
    print(f"\nAccuracy log updated: {ACCURACY_LOG_PATH.resolve()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python daily_feedback.py <path_to_actual_meter_csv>")
        sys.exit(1)
    run_daily_feedback(sys.argv[1])
