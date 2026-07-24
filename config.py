"""
config.py

Single source of truth for plant details, file paths, and pipeline
settings -- imported by every other file in this project. Centralizing
these here avoids circular imports between test_multi_image.py and the
new feature/ML pipeline files, and means you only ever update plant
details in ONE place.
"""

import os
from pathlib import Path


# ---- API credentials ----
# Keep secrets in the project .env file (which is excluded from version
# control). An environment variable takes precedence, which also supports
# deployments that inject credentials externally.
ENV_FILE_PATH = Path(__file__).resolve().with_name(".env")


def _read_env_value(name: str) -> str:
    """Return a value from .env without requiring an extra dependency."""
    try:
        for line in ENV_FILE_PATH.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == name:
                return value.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return ""


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", _read_env_value("GEMINI_API_KEY")).strip()

# ---- Plant details ----
PLANT_NAME = "SIRMOUR"
PLANT_LAT = 24.56253056
PLANT_LON = 75.09140278

# Rated (nameplate) capacity in MW.
PLANT_CAPACITY_MW = 5.1

# Performance Ratio -- accounts for real-world losses (panel temperature,
# inverter, wiring, soiling, shading, mismatch etc.). 0.75-0.85 is typical
# for a well-maintained plant. Update this once you have your plant's
# actual historical PR.
PERFORMANCE_RATIO = 0.78

# ---- Windy capture settings ----
ZOOM_LEVEL = 11  # calibrated so the screenshot covers ~100km x 100km
VIEWPORT_WIDTH = 1600
VIEWPORT_HEIGHT = 1000

LAYERS = {
    "satellite": "Satellite cloud imagery -- cloud position, density, and movement around the plant",
    "wind": "Wind speed, direction, and gusts",
    "solarpower": "Solar power / solar irradiance layer -- expected solar radiation intensity reaching the ground around the plant",
    "clouds": "Cloud cover layer -- overall cloud coverage and thickness around the plant",
    "rain": "Rain / precipitation layer -- rainfall intensity and coverage around the plant",
}

# ---- Animation video settings ----
RECORD_ANIMATION_VIDEO = True
ANIMATION_LAYER = "satellite"
# Short enough to avoid Windy's animation loop/reversal contaminating
# optical-flow features. This measures motion; it is not the forecast horizon.
ANIMATION_RECORD_SECONDS = 8

# ---- Forecast settings ----
NUM_FORECAST_BLOCKS = 8       # 8 x 15 min = next 2 hours
BLOCK_MINUTES = 15

# ---- Capture schedule ----
# Instead of running on a fixed interval, capture only at these fixed
# times each day (24h "HH:MM"), covering morning to evening.
CAPTURE_TIMES = ["06:45", "08:15", "09:45", "11:15", "12:45", "14:15", "15:45"]

# ---- Paths ----
STORAGE_STATE_PATH = Path("windy_login.json")
SCREENSHOT_DIR = Path("windy_screenshots") / f"{PLANT_LAT}_{PLANT_LON}"
VIDEO_DIR = Path("windy_videos")
PREDICTIONS_DIR = Path("energy_predictions")
FEATURES_LOG_DIR = Path("features_log")
MODELS_DIR = Path("models")
ACCURACY_REPORTS_DIR = Path("accuracy_reports")
HISTORIC_CASES_DIR = Path("historic_cases")

# ---- Daily actuals feedback automation ----
# Drop the plant's actual meter-export CSV here at the end of each day.
# Every pipeline run (see daily_feedback.process_actuals_inbox(), called
# from run_pipeline.py) automatically merges any new file here into
# historic_cases/merged_scada_data.csv, compares that day's predictions
# against the actuals, and archives the raw file into the "processed"
# subfolder so it's never re-processed.
ACTUALS_INBOX_DIR = Path("daily_actuals_inbox")
ACTUALS_INBOX_PROCESSED_DIR = ACTUALS_INBOX_DIR / "processed"

# Rolling day-level accuracy/pattern context fed into the LLM prompt (see
# llm_predictor.py) -- keeps only the most recent CONTEXT_WINDOW_DAYS days,
# dropping the oldest each time a new day is added.
PREDICTION_CONTEXT_PATH = Path("prediction_context") / f"{PLANT_NAME}_context.json"
CONTEXT_WINDOW_DAYS = 3

# ---- Case-Based Reasoning retrieval ----
# These weights express the relative importance of visual conditions and
# solar position when comparing a new situation with past feature rows.
# Values are applied after per-column z-score normalization.
CBR_TOP_K = 8
CBR_FEATURE_WEIGHTS = {
    "solar_elevation_deg": 2.5,
    "minute_of_day": 1.5,
    "clouds_bright_pixel_pct": 2.0,
    "satellite_bright_pixel_pct": 2.0,
    "motion_coverage_end_pct": 1.8,
    "motion_score": 1.3,
    "motion_directional_consistency": 0.8,
    "motion_direction_deg": 1.2,
    "rain_bright_pixel_pct": 1.0,
    "solarpower_bright_pixel_pct": 1.0,
}

for _dir in (SCREENSHOT_DIR, VIDEO_DIR, PREDICTIONS_DIR, FEATURES_LOG_DIR, MODELS_DIR, ACCURACY_REPORTS_DIR,
             HISTORIC_CASES_DIR, ACTUALS_INBOX_DIR, ACTUALS_INBOX_PROCESSED_DIR, PREDICTION_CONTEXT_PATH.parent):
    _dir.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "generation_model.pkl"
