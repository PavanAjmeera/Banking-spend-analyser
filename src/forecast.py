"""
STEP 4 - PREDICT NEXT MONTH'S SPEND

Two simple models, compared honestly:
  1. Linear Regression on the month number (captures a trend)
  2. Baseline: average of the last 3 months (simple, hard to beat)

We BACKTEST: pretend we are at month k, predict month k, compare with reality.
Metric: MAE = average rupee error (easy to explain).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def monthly_spend(df: pd.DataFrame, exclude_ids=None) -> pd.DataFrame:
    """exclude_ids = transaction ids flagged as unusual. We forecast NORMAL spending only,
    otherwise one fraud-like Rs 90,000 payment would distort the prediction."""
    debits = df[df["txn_type"] == "DEBIT"]
    if exclude_ids:
        debits = debits[~debits["txn_id"].isin(exclude_ids)]
    m = debits.groupby("month")["amount"].sum().reset_index()
    m.columns = ["month", "spend"]
    return m


def _linear_predict(values):
    X = np.arange(len(values)).reshape(-1, 1)
    model = LinearRegression().fit(X, values)
    return float(model.predict([[len(values)]])[0])


def _baseline_predict(values):
    return float(np.mean(values[-3:]))


def forecast_next_month(monthly: pd.DataFrame, backtest_months: int = 3):
    values = monthly["spend"].to_numpy()
    if len(values) < 5:   # edge case: not enough history
        return None

    errors = {"linear": [], "baseline": []}
    for i in range(len(values) - backtest_months, len(values)):
        actual = values[i]
        errors["linear"].append(abs(_linear_predict(values[:i]) - actual))
        errors["baseline"].append(abs(_baseline_predict(values[:i]) - actual))

    result = {
        "next_month": str(pd.Period(monthly["month"].iloc[-1]) + 1),
        "linear": _linear_predict(values),
        "baseline": _baseline_predict(values),
        "mae_linear": float(np.mean(errors["linear"])),
        "mae_baseline": float(np.mean(errors["baseline"])),
    }
    result["best_model"] = "linear" if result["mae_linear"] < result["mae_baseline"] else "baseline"
    result["prediction"] = result[result["best_model"]]
    return result


def category_forecast(df: pd.DataFrame, exclude_ids=None) -> pd.DataFrame:
    """Per-category forecast = average of the last 3 months."""
    debits = df[df["txn_type"] == "DEBIT"]
    if exclude_ids:
        debits = debits[~debits["txn_id"].isin(exclude_ids)]
    pivot = debits.pivot_table(index="month", columns="category", values="amount",
                               aggfunc="sum", fill_value=0)
    out = pivot.tail(3).mean().round(0).sort_values(ascending=False).reset_index()
    out.columns = ["category", "expected_next_month"]
    return out
