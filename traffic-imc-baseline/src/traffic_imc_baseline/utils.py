import numpy as np

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (sMAPE)."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    nonzero_denom = denominator != 0
    if not nonzero_denom.any():
        return 0.0
    
    smape_value = np.mean(
        np.abs(y_true[nonzero_denom] - y_pred[nonzero_denom])
        / denominator[nonzero_denom]
    ) * 100
    return float(smape_value)