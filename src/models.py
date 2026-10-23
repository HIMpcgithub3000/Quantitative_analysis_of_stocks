import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

def run_adf_test(series: pd.Series, name: str = ""):
    """
    Executes Augmented Dickey-Fuller stationarity hypothesis test.
    H0: Unit root present (non-stationary).
    H1: Stationary process.
    """
    clean_series = series.dropna()
    res = adfuller(clean_series, autolag="AIC")
    return {
        "ticker": name,
        "adf_stat": res[0],
        "p_value": res[1],
        "used_lag": res[2],
        "nobs": res[3],
        "crit_1pct": res[4]["1%"],
        "crit_5pct": res[4]["5%"],
        "crit_10pct": res[4]["10%"],
        "is_stationary": res[1] < 0.05
    }

def fit_arima(train_series, order=(3, 1, 2)):
    """
    Fits an ARIMA(p, d, q) model using maximum likelihood estimation.
    """
    model = ARIMA(train_series, order=order)
    model_fit = model.fit()
    return model_fit

def train_gbr_baseline(X_train, y_train, random_state=42):
    """
    Trains baseline Gradient Boosting Regressor with project specifications.
    """
    gbr = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        random_state=random_state
    )
    gbr.fit(X_train, y_train)
    return gbr

def train_gbr_tuned(X_train, y_train, param_grid=None, n_splits=5, random_state=42):
    """
    Performs TimeSeriesSplit cross-validated grid search over hyperparameter space.
    """
    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200],
            "learning_rate": [0.04, 0.05],
            "max_depth": [3, 4],
            "subsample": [0.8, 0.9]
        }
        
    tscv = TimeSeriesSplit(n_splits=n_splits)
    base_gbr = GradientBoostingRegressor(random_state=random_state)
    
    grid = GridSearchCV(
        estimator=base_gbr,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=2
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid
