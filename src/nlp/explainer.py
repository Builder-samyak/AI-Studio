"""
src/nlp/explainer.py
Destroyers | 42174 AI Studio Autumn 2026

Template-based NLP explanation module for breast cancer classification predictions.
Generates plain-English explanations for clinical users based on:
  - Predicted class (Cancer / Non-Cancer)
  - Malignant probability score
  - Classification threshold

Usage:
    from src.nlp.explainer import generate_explanation
    text = generate_explanation("Cancer", 0.87, threshold=0.5)
"""

DISCLAIMER = (
    "This is a decision-support tool only and must not replace "
    "professional medical diagnosis. All predictions should be reviewed "
    "by a qualified healthcare professional."
)


def generate_explanation(
    pred_class: str,
    probability: float,
    threshold: float = 0.5,
    region_info: str = None,
) -> str:
    """
    Generate a plain-English explanation for a breast cancer classification prediction.

    Args:
        pred_class:   Predicted class label — "Cancer" or "Non-Cancer"
        probability:  Malignant probability score between 0 and 1
        threshold:    Classification threshold used (default 0.5)
        region_info:  Optional description of image region (e.g. "upper-left quadrant")

    Returns:
        A plain-English explanation paragraph suitable for display in the clinical UI.
    """
    pct      = round(probability * 100, 1)
    conf_pct = round(max(probability, 1 - probability) * 100, 1)
    region   = f" in the {region_info}" if region_info else ""

    if pred_class == "Cancer" and probability >= 0.80:
        # High-confidence cancer
        explanation = (
            f"The model identified cellular patterns{region} in this histopathology image "
            f"strongly consistent with malignant tissue (malignant probability: {pct}%). "
            f"Key indicators include irregular cell clustering and nuclear enlargement. "
            f"The model has high confidence ({conf_pct}%) in this classification. "
            f"This case has been flagged for urgent clinical review. "
            f"{DISCLAIMER}"
        )

    elif pred_class == "Cancer" and probability >= threshold:
        # Low-to-moderate confidence cancer
        explanation = (
            f"The model detected possible indicators of malignancy{region} in this "
            f"histopathology image (malignant probability: {pct}%). "
            f"The prediction exceeds the classification threshold of {round(threshold*100)}% "
            f"but confidence is moderate ({conf_pct}%). "
            f"Clinical review is recommended to confirm or rule out malignancy. "
            f"{DISCLAIMER}"
        )

    else:
        # Non-cancer
        explanation = (
            f"The model identified cellular patterns{region} in this histopathology image "
            f"consistent with non-cancerous tissue (malignant probability: {pct}%). "
            f"No malignancy indicators were detected above the classification threshold "
            f"of {round(threshold*100)}%. "
            f"{DISCLAIMER}"
        )

    return explanation


def get_confidence_label(probability: float, threshold: float = 0.5) -> str:
    """Return a human-readable confidence label."""
    score = max(probability, 1 - probability)
    if score >= 0.85:
        return "High"
    elif score >= 0.70:
        return "Moderate"
    else:
        return "Low"


def get_flag_status(pred_class: str, probability: float, threshold: float = 0.5) -> bool:
    """Return True if case should be flagged for clinical review."""
    return pred_class == "Cancer" and probability >= threshold
