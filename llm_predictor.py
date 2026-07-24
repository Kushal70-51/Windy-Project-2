"""
llm_predictor.py

The ONLY module in this pipeline that calls an LLM. Its job is narrow
and constrained on purpose: given a physics-based anchor value for each
of the next 8 forecast blocks, plus the most similar past situations
(from similarity_retrieval.py) and their real outcomes, ask the LLM to
ADJUST each anchor (not invent a number from scratch) and explain why.

WHY THIS DESIGN (vs asking the LLM to just "predict the generation"):
    - The physics anchor keeps every prediction physically grounded
      (never wildly off, always respects day/night and rough cloud
      attenuation) even if the LLM's adjustment is unhelpful.
    - Retrieved similar cases give the LLM concrete historical evidence
      ("in similar cloud conditions, actual generation was X% lower/
      higher than this formula predicted") instead of vague reasoning.
    - A single, small, structured JSON response is far more reliable to
      parse and validate than asking for 8 independent numbers with no
      anchor to sanity-check against.

If the LLM call fails entirely (network, rate limit, bad JSON), this
module falls back to the physics anchor values unchanged -- the pipeline
never produces no output just because the LLM step had a problem.
"""

import json
import time

from google import genai

import config


def is_transient_gemini_error(e) -> bool:
    """Same retry-worthy-error check used throughout this project:
    503/UNAVAILABLE, 429/RESOURCE_EXHAUSTED, 500/INTERNAL are worth
    retrying; anything else (bad key, bad request) is not."""
    msg = str(e)
    transient_markers = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL")
    return any(marker in msg for marker in transient_markers)


def _summarize_current_situation(feature_row: dict) -> str:
    """
    Builds a short, human-readable summary of the CURRENT feature row for
    the prompt -- deliberately NOT dumping all 30+ raw feature values, to
    keep the prompt small, cheap, and easy for the LLM to reason over.
    Only the features that are actually meaningful to a person (and to
    the LLM's reasoning) are included.
    """
    lines = []

    elevation = feature_row.get("solar_elevation_deg")
    if elevation is not None:
        lines.append(f"Solar elevation: {elevation} deg")

    direction_deg = feature_row.get("motion_direction_deg")
    if direction_deg is not None:
        lines.append(f"Cloud motion direction (degrees, -1 = stationary/negligible): {direction_deg}")

    motion_score = feature_row.get("motion_score", feature_row.get("motion_speed_kmh"))
    if motion_score is not None:
        lines.append(f"Relative cloud-motion score: {motion_score} (not physical km/h)")

    coverage_end = feature_row.get("motion_coverage_end_pct")
    if coverage_end is not None:
        lines.append(f"Cloud coverage over plant (video-derived): {coverage_end}%")

    for layer in ("clouds", "satellite", "rain", "solarpower", "wind"):
        brightness_key = f"{layer}_bright_pixel_pct"
        if brightness_key in feature_row and feature_row[brightness_key] is not None:
            lines.append(f"{layer.capitalize()} layer bright-pixel %: {feature_row[brightness_key]}")

    return "\n".join(lines) if lines else "(no readable feature summary available)"


def _build_prompt(anchor_predictions: list, feature_row: dict, retrieved_cases_text: str,
                   context_text: str) -> str:
    blocks_text = "\n".join(
        f"{i + 1}. time={p['time']}, physics_anchor_mw={p['anchor_mw']}"
        for i, p in enumerate(anchor_predictions)
    )

    return f"""
You are assisting with short-term solar generation forecasting for a
plant. A deterministic physics formula has ALREADY produced a baseline
("anchor") estimate for each of the next 8 forecast blocks (15-minute
intervals, covering the next 2 hours). Your job is to ADJUST each anchor
value based on the retrieved historical evidence and recent accuracy
patterns below -- NOT to invent a new number independently of them.

Current situation:
{_summarize_current_situation(feature_row)}

{retrieved_cases_text}

{context_text}

Physics anchor values for the next 8 blocks (baseline, before your adjustment):
{blocks_text}

Using ONLY the current situation and the historical evidence and recent
accuracy patterns above, respond with ONLY a raw JSON array -- no
markdown code fences, no prose before or after. The array must have
EXACTLY {len(anchor_predictions)} objects, one per block, IN THIS ORDER,
each with exactly these keys:
  "time": the block's time exactly as given above (string)
  "adjusted_mw": your adjusted generation estimate in MW (a plain number,
      not a string). If you have no reason to adjust a block, just repeat
      its physics_anchor_mw value -- do not adjust without a reason drawn
      from the historical evidence or recent accuracy patterns.
  "confidence": "High", "Medium", or "Low", based on how closely the
      retrieved historical cases match the current situation (few/no
      similar cases, or cases with very different conditions, should
      mean lower confidence).
  "reasoning": one short sentence explaining the adjustment (or lack of
      one), referencing the actual historical evidence or recent accuracy
      patterns above -- do not invent facts not present in the data given
      to you.

Example format (values illustrative only):
[
  {{"time": "2026-07-20 13:15", "adjusted_mw": 2.1, "confidence": "Medium", "reasoning": "Similar cloud coverage on 2026-07-18 showed generation ~8% below the anchor formula."}}
]
"""


def _parse_llm_response(raw_text: str, anchor_predictions: list) -> list:
    """
    Parses the LLM's JSON response, falling back per-block to the
    physics anchor (with a note explaining why) if parsing fails or a
    block is missing/malformed -- so one bad response never loses the
    whole run's predictions.
    """
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        data = json.loads(text)
    except Exception as e:
        print(f"  [WARN] Could not parse LLM JSON response ({e}); using physics anchor for all blocks.")
        data = []

    by_time = {}
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "time" in item:
                by_time[item["time"]] = item

    results = []
    for anchor in anchor_predictions:
        item = by_time.get(anchor["time"])
        if item and "adjusted_mw" in item:
            try:
                adjusted_mw = float(item["adjusted_mw"])
            except (TypeError, ValueError):
                adjusted_mw = anchor["anchor_mw"]
                item = None
        else:
            adjusted_mw = anchor["anchor_mw"]

        results.append({
            "time": anchor["time"],
            "block_number": anchor["block_number"],
            "anchor_mw": anchor["anchor_mw"],
            "llm_mw": adjusted_mw,
            "confidence": (item or {}).get("confidence", "Low"),
            "reasoning": (item or {}).get(
                "reasoning",
                "LLM adjustment unavailable for this block -- using physics anchor unchanged."
            ),
        })
    return results


def predict_with_llm(anchor_predictions: list, feature_row: dict, retrieved_cases_text: str,
                      context_text: str = "") -> list:
    """
    Main entry point.

    anchor_predictions: list of {"time": ..., "block_number": ..., "anchor_mw": ...}
        for the 8 upcoming forecast blocks (from physics_anchor.py).
    feature_row: the current (most recent capture's) feature dict.
    retrieved_cases_text: output of similarity_retrieval.format_cases_for_prompt().
    context_text: output of daily_feedback.format_context_for_prompt() -- the
        rolling last few days' error/pattern analysis from real meter data.

    Returns a list of dicts, one per block:
        {"time", "block_number", "anchor_mw", "llm_mw", "confidence", "reasoning"}

    If the LLM is unavailable or fails after retries, every block falls
    back to llm_mw == anchor_mw with confidence "Low" and an explanatory
    reasoning string -- the pipeline always produces a full set of
    predictions.
    """
    if not config.GEMINI_API_KEY:
        print("  [WARN] GEMINI_API_KEY not set -- skipping LLM adjustment, using physics anchor for all blocks.")
        return _parse_llm_response("[]", anchor_predictions)

    client = genai.Client(api_key=config.GEMINI_API_KEY, vertexai=False)
    prompt = _build_prompt(anchor_predictions, feature_row, retrieved_cases_text, context_text)

    max_retries = 4
    base_delay = 5
    raw_text = ""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=[prompt],
            )
            raw_text = (response.text or "").strip()
            break
        except Exception as e:
            if is_transient_gemini_error(e) and attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                print(f"  [WARN] Gemini API busy (attempt {attempt}/{max_retries}): {e}")
                print(f"  Retrying in {delay}s...")
                time.sleep(delay)
                continue
            print(f"  [WARN] Gemini call failed ({e}) -- using physics anchor for all blocks.")
            raw_text = ""
            break

    return _parse_llm_response(raw_text, anchor_predictions)
