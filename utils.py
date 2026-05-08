"""
utils.py
========
Shared helper functions used across the project.
"""

import numpy as np
import pandas as pd


def format_transaction(input_dict: dict) -> np.ndarray:
    """
    Convert a dict of {feature_name: value} into a 2-D numpy array
    ready for model.predict().

    Parameters
    ----------
    input_dict : dict
        Keys should match the 29 features (V1–V28 + Amount_scaled).

    Returns
    -------
    np.ndarray  shape (1, 29)
    """
    return np.array(list(input_dict.values())).reshape(1, -1)


def get_risk_label(probability: float) -> tuple[str, str]:
    """
    Map fraud probability to a human-readable risk label + colour.

    Returns
    -------
    (label, colour_hex)
    """
    if probability < 0.3:
        return "✅ Low Risk",    "#2ECC71"
    elif probability < 0.6:
        return "⚠️ Medium Risk", "#F39C12"
    else:
        return "🚨 High Risk — FRAUD DETECTED", "#E74C3C"


def generate_dummy_transaction(fraud: bool = False, seed: int = 0) -> dict:
    """
    Generate a synthetic transaction for demo purposes.
    Uses rough statistical properties of the Kaggle dataset.
    """
    rng = np.random.default_rng(seed)
    if fraud:
        # Fraud transactions tend to have extreme V-values
        v_vals = rng.normal(loc=0, scale=3, size=28)
    else:
        v_vals = rng.normal(loc=0, scale=1, size=28)

    amount = float(rng.uniform(1, 500)) if not fraud else float(rng.uniform(1, 200))
    keys   = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    values = list(v_vals) + [amount]
    return dict(zip(keys, values))


def describe_prediction(prediction: int, probability: float) -> str:
    label, _ = get_risk_label(probability)
    return (
        f"Prediction  : {'FRAUD' if prediction == 1 else 'LEGITIMATE'}\n"
        f"Probability : {probability:.4f} ({probability*100:.2f}%)\n"
        f"Risk Level  : {label}"
    )
