#!/usr/bin/env python3
"""
Quantitative Analysis of Stocks - Main Execution Pipeline
Author: Himanshu Sharma
Repository: https://github.com/HIMpcgithub3000/Quantitative_analysis_of_stocks
"""

import os
import sys
import numpy as np
import pandas as pd

from src.data_loader import load_and_clean_data
from src.features import engineer_features, FEATURE_COLUMNS
from src.models import run_adf_test, fit_arima, train_gbr_baseline, train_gbr_tuned
from src.evaluate import evaluate_predictions

def run_pipeline():
    print("=" * 80)
    print(" 📈 QUANTITATIVE ANALYSIS & FORECASTING OF MEGA-CAP EQUITIES")
    print("=" * 80)
    
    # 1. Load Data
    data_path = "data/combined_ohlc.csv"
    if not os.path.exists(data_path):
        print("Loading from raw CSVs...")
        data = load_and_clean_data(csv_dir="data/csv", save_path=data_path)
    else:
        data = pd.read_csv(data_path)
        data["Date"] = pd.to_datetime(data["Date"])
        print(f"Loaded {len(data)} observations across tickers: {sorted(data['Ticker'].unique().tolist())}")
        
    # 2. Stationarity (ADF Tests)
    print("\n" + "-" * 80)
    print("1. Augmented Dickey-Fuller (ADF) Stationarity Analysis")
    print("-" * 80)
    
    adf_summary = []
    price_pivot = data.pivot_table(index="Date", columns="Ticker", values="Close", aggfunc="last")
    log_returns = np.log(price_pivot / price_pivot.shift(1)).dropna()
    
    for ticker in sorted(data["Ticker"].unique()):
        raw_res = run_adf_test(price_pivot[ticker], name=ticker)
        ret_res = run_adf_test(log_returns[ticker], name=ticker)
        adf_summary.append({
            "Ticker": ticker,
            "Price ADF Stat": f"{raw_res['adf_stat']:.2f}",
            "Price p-value": f"{raw_res['p_value']:.4f}",
            "LogReturn ADF Stat": f"{ret_res['adf_stat']:.2f}",
            "LogReturn p-value": f"{ret_res['p_value']:.4e}",
            "Stationary": "Yes (LogRet)" if ret_res["is_stationary"] else "No"
        })
    print(pd.DataFrame(adf_summary).to_string(index=False))
    
    # 3. Feature Engineering
    print("\n" + "-" * 80)
    print("2. Quantitative Feature Engineering")
    print("-" * 80)
    df_feat = engineer_features(data)
    ml_data = df_feat.dropna(subset=FEATURE_COLUMNS + ["target"]).copy()
    print(f"Generated features: {FEATURE_COLUMNS}")
    print(f"Total ML samples after dropna: {len(ml_data)}")
    
    # 4. Econometric Benchmark: ARIMA(3,1,2) on AAPL
    print("\n" + "-" * 80)
    print("3. Econometric Baseline: ARIMA(3, 1, 2) on AAPL (1-Year Out-of-Sample)")
    print("-" * 80)
    aapl_series = data[data["Ticker"] == "AAPL"].sort_values("Date").set_index("Date")["Close"]
    aapl_train = aapl_series.iloc[:-252]
    aapl_test = aapl_series.iloc[-252:]
    
    arima_fit = fit_arima(aapl_train.values, order=(3, 1, 2))
    arima_forecast = arima_fit.forecast(steps=len(aapl_test))
    
    arima_metrics = evaluate_predictions(aapl_test.values, arima_forecast)
    print(f"ARIMA(3,1,2) AIC: {arima_fit.aic:.2f} | BIC: {arima_fit.bic:.2f}")
    print(f"ARIMA Out-of-Sample -> RMSE: {arima_metrics['RMSE']:.2f} | MAE: {arima_metrics['MAE']:.2f} | MAPE: {arima_metrics['MAPE (%)']:.2f}%")
    
    # 5. ML Models: Cross-Asset Generalization Test (Train on AAPL, AMZN, GOOGL, MSFT -> Test on TSLA)
    print("\n" + "-" * 80)
    print("4. Cross-Asset Generalization: Train 4 Stocks -> Test Unseen Asset (TSLA)")
    print("-" * 80)
    split_idx = int(len(ml_data) * 0.8)
    train_data = ml_data.iloc[:split_idx]
    test_data = ml_data.iloc[split_idx:]
    
    X_train, y_train = train_data[FEATURE_COLUMNS], train_data["target"]
    X_test, y_test = test_data[FEATURE_COLUMNS], test_data["target"]
    y_test_base = test_data["close_lag"]
    
    # Baseline GBR
    gbr_base = train_gbr_baseline(X_train, y_train)
    y_pred_base = gbr_base.predict(X_test)
    base_metrics = evaluate_predictions(y_test, y_pred_base, y_base=y_test_base)
    
    # Tuned GBR (Hyperparameter Grid Search)
    print("Running TimeSeriesSplit Hyperparameter Grid Search...")
    best_gbr, best_params, _ = train_gbr_tuned(X_train, y_train, n_splits=5)
    y_pred_tuned = best_gbr.predict(X_test)
    tuned_metrics = evaluate_predictions(y_test, y_pred_tuned, y_base=y_test_base)
    print(f"Best Hyperparameters: {best_params}")
    
    benchmark_df = pd.DataFrame({
        "Model": ["ARIMA(3,1,2) (AAPL 1-yr)", "Baseline GBR (TSLA Unseen)", "Tuned GBR (TSLA Unseen)"],
        "RMSE": [arima_metrics["RMSE"], base_metrics["RMSE"], tuned_metrics["RMSE"]],
        "MAE": [arima_metrics["MAE"], base_metrics["MAE"], tuned_metrics["MAE"]],
        "MAPE (%)": [arima_metrics["MAPE (%)"], base_metrics["MAPE (%)"], tuned_metrics["MAPE (%)"]],
        "Directional Acc (%)": ["N/A", f"{base_metrics.get('Directional Accuracy (%)', 0):.2f}%", f"{tuned_metrics.get('Directional Accuracy (%)', 0):.2f}%"]
    })
    print("\nBenchmark Results Summary:")
    print(benchmark_df.to_string(index=False))
    
    # 6. Temporal Multi-Asset Backtest
    print("\n" + "-" * 80)
    print("5. Temporal Walk-Forward Multi-Asset Backtest (Train < 2022-03-18 | Test >= 2022-03-18)")
    print("-" * 80)
    dates = sorted(ml_data["Date"].unique())
    cutoff_date = dates[int(len(dates) * 0.8)]
    train_temporal = ml_data[ml_data["Date"] < cutoff_date]
    test_temporal = ml_data[ml_data["Date"] >= cutoff_date]
    
    gbr_temporal = train_gbr_baseline(train_temporal[FEATURE_COLUMNS], train_temporal["target"])
    temp_preds = gbr_temporal.predict(test_temporal[FEATURE_COLUMNS])
    temp_overall = evaluate_predictions(test_temporal["target"], temp_preds, y_base=test_temporal["close_lag"])
    
    temp_results = []
    for ticker in sorted(test_temporal["Ticker"].unique()):
        sub = test_temporal[test_temporal["Ticker"] == ticker]
        p = gbr_temporal.predict(sub[FEATURE_COLUMNS])
        m = evaluate_predictions(sub["target"], p, y_base=sub["close_lag"])
        temp_results.append({
            "Ticker": ticker,
            "Observations": len(sub),
            "RMSE": f"{m['RMSE']:.2f}",
            "MAE": f"{m['MAE']:.2f}",
            "MAPE (%)": f"{m['MAPE (%)']:.2f}%",
            "Dir Acc (%)": f"{m.get('Directional Accuracy (%)', 0):.2f}%"
        })
    temp_results.append({
        "Ticker": "OVERALL",
        "Observations": len(test_temporal),
        "RMSE": f"{temp_overall['RMSE']:.2f}",
        "MAE": f"{temp_overall['MAE']:.2f}",
        "MAPE (%)": f"{temp_overall['MAPE (%)']:.2f}%",
        "Dir Acc (%)": f"{temp_overall.get('Directional Accuracy (%)', 0):.2f}%"
    })
    print(pd.DataFrame(temp_results).to_string(index=False))
    print("\n" + "=" * 80)
    print(" Pipeline execution finished successfully!")
    print("=" * 80)

if __name__ == "__main__":
    run_pipeline()
