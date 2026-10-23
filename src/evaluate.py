import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_predictions(y_true, y_pred, y_base=None):
    """
    Computes standard institutional error metrics and directional accuracy.
    y_true: Ground truth price values (array-like)
    y_pred: Predicted price values (array-like)
    y_base: Previous day prices (Close_{t}) for directional accuracy computation
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mae = mean_absolute_error(y_t, y_p)
    mape = np.mean(np.abs((y_t - y_p) / y_t)) * 100
    r2 = r2_score(y_t, y_p)
    
    results = {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE (%)": float(mape),
        "R2": float(r2)
    }
    
    if y_base is not None:
        y_b = np.asarray(y_base)
        actual_direction = np.sign(y_t - y_b)
        pred_direction = np.sign(y_p - y_b)
        # Exclude zero return days
        non_zero = actual_direction != 0
        if np.sum(non_zero) > 0:
            da = np.mean(actual_direction[non_zero] == pred_direction[non_zero]) * 100
            results["Directional Accuracy (%)"] = float(da)
            
    return results

def format_metrics_table(metrics_dict):
    """
    Formats dictionary of model metrics into a clean DataFrame table.
    """
    df = pd.DataFrame.from_dict(metrics_dict, orient="index")
    return df
