"""
run_pipeline.py

REPLACES analyze_with_gemini() from the old pipeline, and completes the
move to the new hybrid architecture:

    Screenshot capture ----> Image feature extraction -----\\
                                                              >-- Combine features -> Physics anchor
    Satellite video ----> Video processing (optical flow) --/                            |
                                                                                            v
                                                                          Similarity retrieval (top-K
                                                                          similar past cases from the
                                                                          case store / features_log.csv)
                                                                                            |
                                                                                            v
                                                                          LLM reasoning (adjusts anchor
                                                                          using retrieved evidence,
                                                                          explains why)
                                                                                            |
                                                                                            v
                                                                          Validator (range/deviation/
                                                                          smoothness safety checks)
                                                                                            |
                                                                                            v
                                                                          Store prediction (features_log.csv
                                                                          becomes tomorrow's case store)

Only ONE LLM call happens per run, covering all 8 forecast blocks at
once (not one call per block) -- this keeps cost/latency reasonable.

Called from test_multi_image.py's run_once(), right after screenshots +
video have been captured.
"""

import datetime

import config
import image_feature_extraction
import video_motion_features
import time_features
import feature_builder
import physics_anchor
import similarity_retrieval
import llm_predictor
import validator
import prediction_store
import daily_feedback


def run_prediction_pipeline(image_map: dict, video_path):
    """
    image_map: {filepath: description} from capture_all_layers()
    video_path: Path to the recorded/trimmed video, or None if recording failed
    """
    # Make newly available SCADA outcomes usable for retrieval without a
    # separate manual join step. Only timestamp-matched feature rows are
    # enriched, so future forecasts can never leak their own outcomes.
    synced_actuals = daily_feedback.sync_historic_case_actuals()
    if synced_actuals:
        print(f"\nSynced {synced_actuals} historical SCADA actual(s) into the case store.")

    # Pick up any actual-meter CSV the company dropped in daily_actuals_inbox/
    # since the last run: merge it into the case store, run error/pattern
    # analysis for that day, and fold it into the rolling prediction context.
    analyzed_dates = daily_feedback.process_actuals_inbox()
    if analyzed_dates:
        print(f"\nProcessed new actual-meter data from the inbox for: {', '.join(analyzed_dates)}")

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

    # ---- Phase 1: build feature rows + physics anchor for all 8 blocks ----
    print("\nBuilding feature rows and physics anchor for each forecast block...")
    block_times = time_features.get_block_times(datetime.datetime.now())

    feature_rows_by_time = {}
    anchor_predictions = []
    feature_columns = None

    for block_time in block_times:
        block_time_feats = time_features.compute_time_features(block_time)
        feature_row = feature_builder.combine_features(motion_features, image_features, block_time_feats)

        if feature_columns is None:
            feature_columns = feature_builder.get_feature_columns(feature_row)

        anchor_mw = physics_anchor.calculate_anchor_mw(feature_row)
        block_number = time_features.block_number_for_time(block_time)
        time_label = block_time.strftime("%Y-%m-%d %H:%M")

        feature_rows_by_time[time_label] = feature_row
        anchor_predictions.append({
            "time": time_label,
            "block_number": block_number,
            "anchor_mw": anchor_mw,
        })
        print(f"  Block {block_number} ({time_label}): physics anchor = {anchor_mw} MW")

    # image/motion features are identical across all 8 blocks in one run
    # (only time changes) -- the first block's row is a fair
    # representative of "the current situation" for retrieval + the LLM.
    current_feature_row = feature_rows_by_time[anchor_predictions[0]["time"]]

    # ---- Phase 2: retrieve similar past cases from the case store ----
    print("\nRetrieving similar past cases from the case store...")
    retrieved_cases = similarity_retrieval.get_top_k_similar_cases(
        current_feature_row, k=config.CBR_TOP_K, exclude_time=anchor_predictions[0]["time"],
    )
    retrieved_cases_text = similarity_retrieval.format_cases_for_prompt(retrieved_cases)
    print(f"  {retrieved_cases_text}")

    # ---- Phase 3: LLM adjusts the anchor using the retrieved evidence ----
    print("\nAsking LLM to adjust physics anchor using retrieved evidence...")
    context_text = daily_feedback.format_context_for_prompt()
    llm_predictions = llm_predictor.predict_with_llm(
        anchor_predictions, current_feature_row, retrieved_cases_text, context_text,
    )

    # ---- Phase 4: validate (range/deviation/smoothness safety checks) ----
    print("\nValidating LLM-adjusted predictions...")
    validated_predictions = validator.validate_predictions(llm_predictions)
    for p in validated_predictions:
        flag = " [ADJUSTED BY VALIDATOR]" if p["was_adjusted"] else ""
        print(f"  Block {p['block_number']} ({p['time']}): anchor={p['anchor_mw']} MW -> "
              f"final={p['validated_mw']} MW (confidence={p['confidence']}){flag}")
        print(f"    Reasoning: {p['reasoning']}")
        if p["was_adjusted"]:
            print(f"    Validator note: {p['adjustment_note']}")

    # ---- Phase 5: store predictions + case store (features_log) ----
    generation_rows = []
    features_log_rows = []
    for p in validated_predictions:
        final_mw = p["validated_mw"]
        final_kw = round(final_mw * 1000, 1)
        feature_row = feature_rows_by_time[p["time"]]

        generation_rows.append((p["block_number"], p["time"], final_mw, final_kw))
        features_log_rows.append((p["block_number"], p["time"], feature_row, final_mw))

    csv_path = prediction_store.save_generation_csv(generation_rows)
    print(f"\nEnergy generation predictions saved to: {csv_path.resolve()}")

    features_log_path = prediction_store.save_features_log(features_log_rows, feature_columns)
    print(f"Feature rows logged to: {features_log_path.resolve()} "
          f"(this is the case store that similarity_retrieval.py searches, "
          f"and that daily_feedback.py enriches with actual generation each evening)")
