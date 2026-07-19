"""
feature_builder.py

Combines the three feature sources into ONE flat, ML-ready row per
forecast block:
  - video motion features   (from video_motion_features.analyze_video)
  - image features           (from image_feature_extraction.extract_image_features)
  - time / solar features    (from time_features.compute_time_features, differs
                               per block since each block is a different future time)

Categorical values (compass direction, coverage trend) are encoded to
numbers here, since ML models like XGBoost/LightGBM need numeric input.
"""

# Compass direction -> degrees (0 = North, clockwise), so the ML model
# gets a continuous numeric value instead of a text label.
_DIRECTION_TO_DEGREES = {
    "North": 0, "Northeast": 45, "East": 90, "Southeast": 135,
    "South": 180, "Southwest": 225, "West": 270, "Northwest": 315,
    "negligible / stationary": -1,  # sentinel: "no meaningful motion"
}

_TREND_TO_NUMBER = {
    "increasing": 1,
    "stable": 0,
    "decreasing": -1,
}


def _default_motion_features() -> dict:
    """Used when no video was recorded/processed, so the pipeline can
    still run (with motion features neutrally filled in) instead of
    crashing."""
    return {
        "avg_direction": "negligible / stationary",
        "avg_speed_kmh": 0.0,
        "coverage_start_pct": 0.0,
        "coverage_end_pct": 0.0,
        "coverage_trend": "stable",
        "frame_count": 0,
    }


def encode_motion_features(motion_features: dict | None) -> dict:
    mf = motion_features or _default_motion_features()
    return {
        "motion_direction_deg": _DIRECTION_TO_DEGREES.get(mf.get("avg_direction"), -1),
        "motion_speed_kmh": mf.get("avg_speed_kmh", 0.0) or 0.0,
        "motion_coverage_start_pct": mf.get("coverage_start_pct", 0.0) or 0.0,
        "motion_coverage_end_pct": mf.get("coverage_end_pct", 0.0) or 0.0,
        "motion_coverage_trend": _TREND_TO_NUMBER.get(mf.get("coverage_trend"), 0),
        "motion_frame_count": mf.get("frame_count", 0) or 0,
    }


def combine_features(motion_features: dict | None, image_features: dict, time_features: dict) -> dict:
    """
    Returns one flat dict combining all three sources for a SINGLE
    forecast block. `image_features` and `motion_features` are the same
    for every block in a run (they only change once you re-capture),
    `time_features` differs per block (since each block is a different
    future timestamp).
    """
    row = {}
    row.update(encode_motion_features(motion_features))
    row.update(image_features)
    row.update(time_features)
    return row


def get_feature_columns(sample_row: dict) -> list:
    """
    Returns a SORTED, stable list of feature column names from a sample
    row. Sorting alphabetically means the column order is always the
    same regardless of dict insertion order -- important so the model
    always sees features in the same order at train time and predict time.
    """
    return sorted(sample_row.keys())
