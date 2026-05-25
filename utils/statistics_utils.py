import numpy as np
import pandas as pd

def compute_metrics(df: pd.DataFrame):
    if df.empty:
        return {
            "mae": 0,
            "rmse": 0,
            "bias": 0,
            "p90_error": 0,
            "count": 0
        }

    errors = df["act_min"] - df["pred_min"]

    metrics = {
        "mae": np.mean(np.abs(errors)),
        "rmse": np.sqrt(np.mean(errors ** 2)),
        "bias": np.mean(errors),
        "p90_error": np.percentile(np.abs(errors), 90),
        "count": len(df)
    }

    return metrics