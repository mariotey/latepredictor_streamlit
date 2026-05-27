import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def compute_metrics(df: pd.DataFrame):
    if df.empty:
        return {
            "mae": 0,
            "mse": 0,
            "rmse": 0,
            "bias": 0,
            "p90_error": 0,
            "count": 0
        }

    y_true = df["act_min"]
    y_pred = df["pred_min"]

    mse = mean_squared_error(y_true, y_pred)

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "bias": np.mean(y_pred - y_true),
        "p90_error": np.percentile(np.abs(y_pred - y_true), 90),
        "count": len(df)
    }
