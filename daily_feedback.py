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
import json
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

# ---- Raw company meter-export columns (the daily_actuals_inbox format) ----
# Same shape as historic_cases/*_SOLAR_INV.csv: TimeStamp + Active Power in
# kW (not MW yet) + raw sensor columns. Update this if the company ever
# changes their export's column names/order.
RAW_METER_COLUMNS = [
    "TimeStamp", "Active Power (kW)", "POA (W/m2)", "GHI (W/m2)",
    "Wind Speed (m/s)", "Wind Direction (DEG.)", "AMB TEMP", "MOD TEMP", "Humidity",
]
MERGED_STORE_PATH = config.HISTORIC_CASES_DIR / "merged_scada_data.csv"


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


def _log_accuracy(metrics: dict, date_str: str = None) -> None:
    """Appends a metrics row to the running accuracy log, so accuracy over
    time can be reviewed later. Defaults to today's date; pass date_str
    explicitly when logging metrics for a day other than today (e.g. a
    company export dropped a day late)."""
    ACCURACY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    row = {"date": date_str or datetime.date.today().isoformat(), **metrics}
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


def _merge_meter_csv_into_store(csv_path: Path) -> set:
    """
    Appends one raw company meter-export CSV (TimeStamp + Active Power in
    kW + sensor columns -- no MW column yet, the same shape the plant
    sends every evening) into historic_cases/merged_scada_data.csv, keyed
    by timestamp so re-dropping the same file twice updates rows instead
    of duplicating them.

    Negative Active Power (kW) readings are clipped to 0 (no generation --
    the same convention already used everywhere else in this project), and
    Active Power (MW) is derived from the clipped kW value.

    Returns the set of calendar-date strings ("YYYY-MM-DD") the file
    touched, so the caller knows which day(s) to analyze.
    """
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != RAW_METER_COLUMNS:
            raise ValueError(
                f"unexpected columns {reader.fieldnames} -- expected exactly {RAW_METER_COLUMNS}"
            )
        new_rows = list(reader)

    merged_header = RAW_METER_COLUMNS + ["Active Power (MW)"]
    existing_by_time = {}
    if MERGED_STORE_PATH.exists():
        with open(MERGED_STORE_PATH, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_by_time[row["TimeStamp"]] = row

    touched_dates = set()
    for row in new_rows:
        kw = max(0.0, float(row["Active Power (kW)"]))
        row = dict(row)
        row["Active Power (kW)"] = str(kw)
        row["Active Power (MW)"] = str(kw / 1000.0)
        existing_by_time[row["TimeStamp"]] = row
        touched_dates.add(row["TimeStamp"][:10])

    sorted_times = sorted(
        existing_by_time.keys(),
        key=lambda t: datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S"),
    )
    with open(MERGED_STORE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=merged_header)
        writer.writeheader()
        for t in sorted_times:
            writer.writerow(existing_by_time[t])

    return touched_dates


def _load_features_log_rows_for_date(date_str: str) -> list:
    """Returns every features_log.csv row for date_str that has both a
    predicted and a (now-synced) actual generation value."""
    path = config.FEATURES_LOG_DIR / f"{config.PLANT_NAME}_features_log.csv"
    if not path.exists():
        return []

    matched = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("Time", "").startswith(date_str):
                continue
            try:
                predicted_mw = float(row.get("Predicted Generation (MW)"))
                actual_mw = float(row.get(ACTUAL_COLUMN_NAME))
            except (TypeError, ValueError):
                continue
            matched.append({
                "time": row["Time"], "predicted_mw": predicted_mw,
                "actual_mw": actual_mw, "row": row,
            })
    return matched


def _time_of_day_bucket(time_label: str) -> str:
    hour = int(time_label[11:13])
    if hour < 10:
        return "morning (before 10:00)"
    if hour < 14:
        return "midday (10:00-14:00)"
    return "afternoon (14:00+)"


def _analyze_day_patterns(date_str: str, matched: list) -> dict:
    """
    Deterministic (no LLM) day-level error + pattern analysis: overall
    MAE/RMSE/MAPE/Bias, bias broken down by time-of-day bucket, and the
    single worst-forecast block with the conditions (engineered features)
    present at that time -- packaged as a compact human-readable summary
    that later gets fed into the LLM prompt as evidence.
    """
    pairs = [(m["predicted_mw"], m["actual_mw"]) for m in matched]
    metrics = _compute_error_metrics(pairs)

    buckets = {}
    for m in matched:
        buckets.setdefault(_time_of_day_bucket(m["time"]), []).append(m["predicted_mw"] - m["actual_mw"])
    bucket_bias = {bucket: round(sum(errs) / len(errs), 3) for bucket, errs in buckets.items()}

    worst = max(matched, key=lambda m: abs(m["predicted_mw"] - m["actual_mw"]))
    notable_bits = []
    for key, label in (
        ("clouds_bright_pixel_pct", "cloud-layer bright-pixel %"),
        ("satellite_bright_pixel_pct", "satellite bright-pixel %"),
        ("motion_coverage_end_pct", "video cloud coverage %"),
        ("motion_direction_deg", "cloud motion direction (deg, -1=stationary)"),
        ("solar_elevation_deg", "solar elevation (deg)"),
    ):
        if worst["row"].get(key):
            notable_bits.append(f"{label}={worst['row'][key]}")

    bias = metrics["bias"]
    bias_direction = "over-forecast" if bias > 0.01 else ("under-forecast" if bias < -0.01 else "roughly balanced")
    mape_str = f"{metrics['mape_pct']}%" if metrics.get("mape_pct") is not None else "n/a"
    bucket_text = "; ".join(
        f"{bucket}: {bucket_bias[bucket]:+} MW" for bucket in sorted(bucket_bias)
    )
    worst_error = worst["predicted_mw"] - worst["actual_mw"]

    summary = (
        f"{date_str}: MAE={metrics['mae']} MW, RMSE={metrics['rmse']} MW, MAPE={mape_str}, "
        f"Bias={bias:+.3f} MW ({bias_direction}). By time of day -- {bucket_text}. "
        f"Worst block at {worst['time']}: predicted {worst['predicted_mw']} MW vs actual "
        f"{worst['actual_mw']} MW (error {worst_error:+.3f} MW)"
        + (f", conditions: {', '.join(notable_bits)}." if notable_bits else ".")
    )

    return {
        "date": date_str,
        "n_matched_blocks": metrics["n_matched_blocks"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "mape_pct": metrics["mape_pct"],
        "bias": bias,
        "bias_direction": bias_direction,
        "time_of_day_bias": bucket_bias,
        "worst_block": {
            "time": worst["time"], "predicted_mw": worst["predicted_mw"],
            "actual_mw": worst["actual_mw"], "error_mw": round(worst_error, 3),
        },
        "summary": summary,
    }


def _load_context() -> list:
    if not config.PREDICTION_CONTEXT_PATH.exists():
        return []
    try:
        return json.loads(config.PREDICTION_CONTEXT_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _add_day_to_context(entry: dict) -> None:
    """Adds/replaces today's entry and keeps only the most recent
    config.CONTEXT_WINDOW_DAYS days, dropping the oldest."""
    entries = [e for e in _load_context() if e["date"] != entry["date"]]
    entries.append(entry)
    entries = sorted(entries, key=lambda e: e["date"])[-config.CONTEXT_WINDOW_DAYS:]

    config.PREDICTION_CONTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.PREDICTION_CONTEXT_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def format_context_for_prompt() -> str:
    """Renders the rolling day-level context as compact text for
    llm_predictor.py's prompt. Called every run regardless of whether the
    inbox had new files this run -- it always reflects the last
    CONTEXT_WINDOW_DAYS days of accumulated learnings."""
    entries = _load_context()
    if not entries:
        return "No recent day-level accuracy history is available yet."

    lines = [f"Recent day-level forecast accuracy and patterns (last {len(entries)} day(s), oldest first):"]
    lines += [f"- {e['summary']}" for e in entries]
    lines.append(
        "If a bias direction or time-of-day tendency repeats across these days, factor it into "
        "your adjustment; a pattern seen on only one day is weaker evidence than one repeated "
        "across multiple days."
    )
    return "\n".join(lines)


def process_actuals_inbox() -> list:
    """
    Call once per pipeline run (see run_pipeline.py). Scans
    config.ACTUALS_INBOX_DIR for new company meter-export CSVs. For each
    file found: merges it into historic_cases/merged_scada_data.csv,
    re-syncs actuals into the feature-log case store, runs error/pattern
    analysis for every day the file touched, folds each day's analysis
    into the rolling prediction-context file (capped at
    config.CONTEXT_WINDOW_DAYS days), and archives the file into
    config.ACTUALS_INBOX_PROCESSED_DIR so it is never re-processed.

    Returns the list of date strings analyzed this call (for logging).
    """
    inbox_files = sorted(config.ACTUALS_INBOX_DIR.glob("*.csv"))
    if not inbox_files:
        return []

    analyzed_dates = []
    for csv_path in inbox_files:
        try:
            touched_dates = _merge_meter_csv_into_store(csv_path)
        except (OSError, ValueError, KeyError) as e:
            print(f"  [WARN] Skipping {csv_path.name} in actuals inbox ({e}).")
            continue

        sync_historic_case_actuals()

        for date_str in sorted(touched_dates):
            matched = _load_features_log_rows_for_date(date_str)
            if not matched:
                print(f"  [INFO] {csv_path.name}: no predicted+actual matches for {date_str} yet "
                      f"-- skipping pattern analysis for this date.")
                continue

            entry = _analyze_day_patterns(date_str, matched)
            _add_day_to_context(entry)
            _log_accuracy(
                {k: entry[k] for k in ("n_matched_blocks", "mae", "rmse", "mape_pct", "bias")},
                date_str=date_str,
            )
            print(f"  [OK] Analyzed {date_str}: {entry['summary']}")
            analyzed_dates.append(date_str)

        config.ACTUALS_INBOX_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        archived_path = config.ACTUALS_INBOX_PROCESSED_DIR / csv_path.name
        if archived_path.exists():
            timestamp = datetime.datetime.now().strftime("%H%M%S")
            archived_path = config.ACTUALS_INBOX_PROCESSED_DIR / f"{csv_path.stem}_{timestamp}{csv_path.suffix}"
        csv_path.rename(archived_path)
        print(f"  Archived {csv_path.name} -> {archived_path}")

    return analyzed_dates


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
