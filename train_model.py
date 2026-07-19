"""
train_model.py

STANDALONE script -- run this once you have enough accumulated data:
    python train_model.py path/to/actual_meter.csv [timestamp_column] [generation_column]

What it does:
1. Reads features_log/<PLANT>_features_log.csv (built up automatically
   every time test_multi_image.py runs -- every 15-min block's full
   feature row gets logged there).
2. Reads your actual meter/SCADA export.
3. Joins them on timestamp -> builds (X, y) training pairs.
4. Trains an XGBoost regressor (falls back to LightGBM, then a plain
   scikit-learn RandomForest, if XGBoost isn't installed).
5. Saves the trained model to config.MODEL_PATH -- ml_forecast_model.py
   picks it up automatically on the next run, no code changes needed.

Requirements:
    pip install xgboost scikit-learn joblib pandas
    (or: pip install lightgbm scikit-learn joblib pandas)

NOTE: you need a reasonable number of matched (features, actual) rows
before this is worth running -- a few days of 15-min data is a bare
minimum; a few weeks to a few months is much better, and ideally
spanning different seasons/weather conditions.
"""

import csv
import datetime
import sys
from pathlib import Path

import config
from accuracy_tracker import DEFAULT_TIMESTAMP_COLUMN, DEFAULT_GENERATION_COLUMN

try:
    import joblib
except ImportError:
    raise SystemExit("joblib is required: pip install joblib")


def _load_features_log():
    csv_path = config.FEATURES_LOG_DIR / f"{config.PLANT_NAME}_features_log.csv"
    if not csv_path.exists():
        raise SystemExit(
            f"No features log found at {csv_path}. Run test_multi_image.py a few "
            f"times first so features accumulate, then come back to this script."
        )

    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        feature_columns = [c for c in reader.fieldnames if c not in ("Block", "Time", "Predicted Generation (MW)")]
        for row in reader:
            try:
                dt = datetime.datetime.strptime(row["Time"], "%Y-%m-%d %H:%M")
            except ValueError:
                continue
            features = {}
            valid = True
            for col in feature_columns:
                try:
                    features[col] = float(row[col])
                except (ValueError, TypeError):
                    valid = False
                    break
            if valid:
                rows.append((dt, features))
    return rows, feature_columns


def _load_actual(csv_path, timestamp_column, generation_column):
    out = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if timestamp_column not in reader.fieldnames or generation_column not in reader.fieldnames:
            raise SystemExit(
                f"Columns not found in {csv_path}. Available columns: {reader.fieldnames}"
            )
        for row in reader:
            try:
                dt = datetime.datetime.strptime(row[timestamp_column], "%Y-%m-%d %H:%M")
                out[dt] = float(row[generation_column])
            except (ValueError, KeyError):
                continue
    return out


def _get_regressor():
    """Tries XGBoost first, then LightGBM, then RandomForest -- whichever
    is installed. Keeps this script usable even with a minimal install."""
    try:
        from xgboost import XGBRegressor
        print("Using XGBoost.")
        return XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor
        print("Using LightGBM.")
        return LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
    except ImportError:
        pass
    from sklearn.ensemble import RandomForestRegressor
    print("Neither XGBoost nor LightGBM found -- using scikit-learn RandomForestRegressor instead.")
    return RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)


def train(actual_csv, timestamp_column=DEFAULT_TIMESTAMP_COLUMN, generation_column=DEFAULT_GENERATION_COLUMN):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error

    feature_rows, feature_columns = _load_features_log()
    actual = _load_actual(Path(actual_csv), timestamp_column, generation_column)

    X, y = [], []
    for dt, features in feature_rows:
        if dt in actual:
            X.append([features[col] for col in feature_columns])
            y.append(actual[dt])

    print(f"Matched {len(X)} training rows out of {len(feature_rows)} logged feature rows.")
    if len(X) < 20:
        raise SystemExit(
            f"Only {len(X)} matched rows -- that's too few to train a reliable model. "
            f"Keep running test_multi_image.py and collecting actual meter data, "
            f"then try again once you have at least a few hundred matched rows."
        )

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = _get_regressor()
    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_predictions)
    print(f"Validation MAE: {mae:.4f} MW (on {len(X_val)} held-out rows)")

    bundle = {"model": model, "feature_columns": feature_columns}
    joblib.dump(bundle, config.MODEL_PATH)
    print(f"Model saved to: {config.MODEL_PATH.resolve()}")
    print("ml_forecast_model.py will pick this up automatically on the next run.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_model.py <actual_meter_csv> [timestamp_column] [generation_column]")
        sys.exit(1)

    actual_csv_path = sys.argv[1]
    ts_col = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TIMESTAMP_COLUMN
    gen_col = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_GENERATION_COLUMN

    train(actual_csv_path, ts_col, gen_col)
