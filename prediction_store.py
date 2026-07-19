"""
prediction_store.py

Saves two CSVs, both using UPDATE-OR-APPEND logic keyed on "Time" (same
approach as before): if a row for that time already exists, it's
updated with the latest prediction; if not, it's appended. Both files
stay sorted chronologically.

1. energy_predictions/<PLANT>_energy_generation.csv
   -> the final, human-facing output: Block, Time, MW, kW.

2. features_log/<PLANT>_features_log.csv
   -> Block, Time, every raw feature value used, AND the predicted MW.
   This is what train_model.py will later join against your actual
   meter/SCADA data to build a training set -- so keep this file
   around, don't delete it.
"""

import csv
import datetime

import config


def _update_or_append(csv_path, header, rows_by_time):
    """Shared read-merge-sort-write logic for both CSV files."""
    existing_by_time = {}
    if csv_path.exists():
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
            for row in reader:
                if row:
                    existing_by_time[row[1]] = row  # row[1] = Time column

    existing_by_time.update(rows_by_time)

    def _parse_time(time_label):
        return datetime.datetime.strptime(time_label, "%Y-%m-%d %H:%M")

    sorted_times = sorted(existing_by_time.keys(), key=_parse_time)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for t in sorted_times:
            writer.writerow(existing_by_time[t])


def save_generation_csv(rows):
    """
    `rows`: list of (block_number, time_label, generation_mw, generation_kw)
    """
    csv_path = config.PREDICTIONS_DIR / f"{config.PLANT_NAME}_energy_generation.csv"
    header = ["Block", "Time", "Predicted Generation (MW)", "Predicted Generation (kW)"]

    rows_by_time = {}
    for block_number, time_label, mw, kw in rows:
        rows_by_time[time_label] = [str(block_number), time_label, str(mw), str(kw)]

    _update_or_append(csv_path, header, rows_by_time)
    return csv_path


def save_features_log(rows, feature_columns):
    """
    `rows`: list of (block_number, time_label, feature_row_dict, generation_mw)
    `feature_columns`: sorted list of feature names (for a stable column
    order across every call -- see feature_builder.get_feature_columns).
    """
    csv_path = config.FEATURES_LOG_DIR / f"{config.PLANT_NAME}_features_log.csv"
    header = ["Block", "Time"] + feature_columns + ["Predicted Generation (MW)"]

    rows_by_time = {}
    for block_number, time_label, feature_row, mw in rows:
        row = [str(block_number), time_label]
        row += [str(feature_row.get(col, "")) for col in feature_columns]
        row.append(str(mw))
        rows_by_time[time_label] = row

    _update_or_append(csv_path, header, rows_by_time)
    return csv_path
