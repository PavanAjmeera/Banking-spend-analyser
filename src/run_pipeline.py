"""Runs everything in the terminal (no dashboard). Good for quick testing.
Run:  python src/run_pipeline.py
"""
from pathlib import Path
from categorise import load_and_prepare
from anomaly import detect_anomalies, evaluate
from forecast import monthly_spend, forecast_next_month, category_forecast

path = Path(__file__).resolve().parent.parent / "data" / "transactions.csv"
df, log = load_and_prepare(path)
print("CLEANING LOG:", log)
print("\nTOP CATEGORIES:\n", df[df.txn_type == "DEBIT"].groupby("category")["amount"].sum().sort_values(ascending=False))

debits = detect_anomalies(df)
print("\nANOMALY CHECK:", evaluate(debits))
print(debits[debits.is_anomaly][["timestamp", "description", "amount", "anomaly_score"]].head(10))

flagged = set(debits[debits.is_anomaly]["txn_id"])
monthly = monthly_spend(df, exclude_ids=flagged)
print("\nFORECAST (normal spend only):", forecast_next_month(monthly))
print("\nBY CATEGORY:\n", category_forecast(df, exclude_ids=flagged))
