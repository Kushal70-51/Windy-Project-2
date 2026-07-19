"""
run_pipeline.py

This REPLACES analyze_with_gemini() from the old pipeline. No LLM is
called anywhere in here -- it's pure feature-extraction + ML, matching
your diagram:

    Screenshot capture ----> Image feature extraction -----\\
                                                              >-- Combine features -> ML model -> Generation prediction -> Store prediction
    Satellite video ----> Video processing (optical flow) --/

Called from test_multi_image.py's run_once(), right after screenshots +
video have been captured.
"""

import datetime

import config
import image_feature_extraction
import video_motion_features
import time_features
import feature_builder
import ml_forecast_model
import prediction_store


def run_prediction_pipeline(image_map: dict, video_path):
    """
    image_map: {filepath: description} from capture_all_layers()
    video_path: Path to the recorded/trimmed video, or None if recording failed
    """
    print("\nExtracting image features (color + brightness per layer)...")
    image_features = image_feature_extraction.extract_image_features(image_map)
    print(f"  [OK] Extracted {len(image_features)} image-derived features.")

    motion_features = None
    if video_path is not None and video_path.exists():
        print(f"\nExtracting video motion features: {video_path}")
        motion_features = video_motion_features.analyze_video(video_path)
        if motion_features is None:
            print("  [WARN] Video feature extraction failed -- continuing without motion features.")
        else:
            print(f"  [OK] {motion_features['summary_text']}")
    else:
        print("\n[INFO] No video available -- continuing without motion features.")

    print("\nBuilding feature rows for each forecast block and predicting generation...")
    block_times = time_features.get_block_times(datetime.datetime.now())

    generation_rows = []
    features_log_rows = []
    feature_columns = None

    for block_time in block_times:
        block_time_feats = time_features.compute_time_features(block_time)
        feature_row = feature_builder.combine_features(motion_features, image_features, block_time_feats)

        if feature_columns is None:
            feature_columns = feature_builder.get_feature_columns(feature_row)

        generation_mw = ml_forecast_model.predict_generation_mw(feature_row)
        generation_kw = round(generation_mw * 1000, 1)

        block_number = time_features.block_number_for_time(block_time)
        time_label = block_time.strftime("%Y-%m-%d %H:%M")

        generation_rows.append((block_number, time_label, generation_mw, generation_kw))
        features_log_rows.append((block_number, time_label, feature_row, generation_mw))

        print(f"  Block {block_number} ({time_label}): {generation_mw} MW ({generation_kw} kW)")

    csv_path = prediction_store.save_generation_csv(generation_rows)
    print(f"\nEnergy generation predictions saved to: {csv_path.resolve()}")

    features_log_path = prediction_store.save_features_log(features_log_rows, feature_columns)
    print(f"Feature rows logged to: {features_log_path.resolve()} "
          f"(this file is what train_model.py will use later)")
