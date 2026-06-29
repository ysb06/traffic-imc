import numpy as np


def _normalize_metric_array(data: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(data)
    if array.ndim < 2:
        raise ValueError(f"{name} must have at least 2 dimensions: {array.shape}")
    if array.ndim >= 3 and array.shape[-1] == 1:
        array = np.squeeze(array, axis=-1)
    return array


def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_true - y_pred
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
    }


def compute_test_metrics_by_horizon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    missing_mask: np.ndarray | None = None,
    horizons: tuple[int, ...] = (1, 3, 6, 12, 24),
    prefix: str = "test",
) -> dict[str, float | int]:
    y_true_arr = _normalize_metric_array(y_true, "y_true")
    y_pred_arr = _normalize_metric_array(y_pred, "y_pred")
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape: "
            f"{y_true_arr.shape} != {y_pred_arr.shape}"
        )

    if missing_mask is None:
        missing_arr = np.zeros(y_true_arr.shape, dtype=bool)
    else:
        missing_arr = _normalize_metric_array(missing_mask, "missing_mask").astype(bool)
        if missing_arr.shape != y_true_arr.shape:
            raise ValueError(
                f"missing_mask must match y_true shape after normalization: "
                f"{missing_arr.shape} != {y_true_arr.shape}"
            )

    valid_mask = ~missing_arr
    total_points = int(valid_mask.size)
    valid_points = int(valid_mask.sum())
    metrics: dict[str, float | int] = {
        f"{prefix}_valid_points": valid_points,
        f"{prefix}_missing_points": total_points - valid_points,
    }

    if valid_points > 0:
        overall = _calculate_metrics(y_true_arr[valid_mask], y_pred_arr[valid_mask])
        metrics[f"{prefix}_mae"] = overall["mae"]
        metrics[f"{prefix}_rmse"] = overall["rmse"]

    horizon_count = y_true_arr.shape[1]
    for horizon in horizons:
        if horizon <= 0:
            raise ValueError(f"horizons must be positive integers: {horizon}")
        if horizon > horizon_count:
            continue

        horizon_idx = horizon - 1
        suffix = f"h{horizon:02d}"
        y_true_h = y_true_arr[:, horizon_idx, ...]
        y_pred_h = y_pred_arr[:, horizon_idx, ...]
        valid_h = valid_mask[:, horizon_idx, ...]
        total_h = int(valid_h.size)
        valid_count_h = int(valid_h.sum())

        metrics[f"{prefix}_valid_points_{suffix}"] = valid_count_h
        metrics[f"{prefix}_missing_points_{suffix}"] = total_h - valid_count_h

        if valid_count_h == 0:
            continue

        horizon_metrics = _calculate_metrics(y_true_h[valid_h], y_pred_h[valid_h])
        metrics[f"{prefix}_mae_{suffix}"] = horizon_metrics["mae"]
        metrics[f"{prefix}_rmse_{suffix}"] = horizon_metrics["rmse"]

    return metrics
