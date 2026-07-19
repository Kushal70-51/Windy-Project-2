"""
ml_forecast_model.py

Wraps the actual ML model (XGBoost or LightGBM) that predicts generation
(MW) from the combined feature row. This is what REPLACES Gemini in the
pipeline -- no LLM is involved in producing the numeric forecast anymore.

IMPORTANT -- "cold start" problem: a model can only be trained once you
have enough (features, actual generation) historical pairs, which takes
time to accumulate (see train_model.py). Until then, this module falls
back to a simple PHYSICS-BASED estimate (solar-elevation-driven clear-sky
proxy, attenuated by cloud coverage) so the pipeline still produces
usable numbers from day one, instead of failing or returning nothing.

Once you've trained a model with train_model.py, it gets picked up
automatically (no code changes needed) -- this module always tries to
load config.MODEL_PATH first, and only uses the fallback if that fails.
"""

import config

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False


_cached_model = None
_model_load_attempted = False


def _load_model():
    """Loads the trained model from disk once, caching it in memory for
    subsequent calls within the same run. Returns None if unavailable."""
    global _cached_model, _model_load_attempted

    if _model_load_attempted:
        return _cached_model
    _model_load_attempted = True

    if not _JOBLIB_AVAILABLE:
        print("  [INFO] joblib not installed -- using physics fallback instead of a trained ML model.")
        return None

    if not config.MODEL_PATH.exists():
        print(f"  [INFO] No trained model found at {config.MODEL_PATH} yet -- "
              f"using physics fallback until you run train_model.py.")
        return None

    try:
        bundle = joblib.load(config.MODEL_PATH)
        _cached_model = bundle  # dict: {"model": ..., "feature_columns": [...]}
        print(f"  [OK] Loaded trained model from {config.MODEL_PATH}")
        return _cached_model
    except Exception as e:
        print(f"  [WARN] Could not load model ({e}) -- using physics fallback.")
        return None


def _fallback_predict_mw(feature_row: dict, capacity_mw: float, performance_ratio: float) -> float:
    """
    Simple physics-ish estimate used until a real model is trained:
      1. Use solar elevation as a rough "clear sky" proxy (0 at night,
         maxing out around solar noon).
      2. Attenuate that by how much cloud is present, blending the
         image-derived brightness/cloud stats with the video motion
         coverage stats.
      3. Scale by plant capacity and performance ratio.
    This will NOT be as accurate as a properly trained model, but keeps
    the pipeline usable from day one.
    """
    elevation = feature_row.get("solar_elevation_deg", 0.0)
    if elevation <= 0:
        return 0.0  # sun below horizon -> no generation

    # Clear-sky proxy: rises with sin(elevation), roughly matching how
    # actual clear-sky irradiance behaves through the day.
    import math
    clear_sky_index = math.sin(math.radians(elevation))
    clear_sky_index = max(0.0, min(1.0, clear_sky_index))

    # Cloud attenuation: average a few cloud-ish signals into one 0-1
    # "how clear is it" factor. Higher bright_pixel_pct / coverage = more
    # cloud = less clear.
    cloud_signals = []
    for key in ("clouds_bright_pixel_pct", "satellite_bright_pixel_pct", "rain_bright_pixel_pct"):
        if feature_row.get(key) is not None:
            cloud_signals.append(feature_row[key] / 100.0)
    motion_cov = feature_row.get("motion_coverage_end_pct")
    if motion_cov is not None:
        cloud_signals.append(motion_cov / 100.0)

    avg_cloud_fraction = sum(cloud_signals) / len(cloud_signals) if cloud_signals else 0.3
    avg_cloud_fraction = max(0.0, min(1.0, avg_cloud_fraction))
    clearness_factor = 1.0 - (0.8 * avg_cloud_fraction)  # heavy cloud can cut ~80%

    generation_mw = capacity_mw * clear_sky_index * clearness_factor * performance_ratio
    return max(0.0, min(capacity_mw, generation_mw))


def predict_generation_mw(feature_row: dict, capacity_mw: float = config.PLANT_CAPACITY_MW,
                           performance_ratio: float = config.PERFORMANCE_RATIO) -> float:
    """
    Main entry point: predicts generation in MW for one forecast block's
    feature row. Uses the trained model if available, otherwise the
    physics fallback above.
    """
    bundle = _load_model()

    if bundle is not None:
        model = bundle["model"]
        feature_columns = bundle["feature_columns"]
        try:
            x = [[feature_row.get(col, 0.0) or 0.0 for col in feature_columns]]
            prediction = float(model.predict(x)[0])
            return max(0.0, min(capacity_mw, prediction))
        except Exception as e:
            print(f"  [WARN] Trained model prediction failed ({e}) -- falling back to physics estimate.")

    return round(_fallback_predict_mw(feature_row, capacity_mw, performance_ratio), 3)
