"""
STEP 3 - UNUSUAL TRANSACTION DETECTION (Isolation Forest)

Idea: Isolation Forest randomly splits the data again and again. Weird points
(rare, extreme) get isolated in very few splits, so they get a high anomaly score.
It is UNSUPERVISED: we do not need labelled fraud examples.

Features we give it:
  - log of the amount (so huge values don't dominate)
  - hour of day
  - amount compared with that category's typical (median) amount
  - is_night flag (12am-5am)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.03):
    """contamination = rough share of transactions we expect to be unusual."""
    debits = df[df["txn_type"] == "DEBIT"].copy()

    if len(debits) < 30:   # edge case: too little data to learn from
        debits["is_anomaly"] = False
        debits["anomaly_score"] = 0.0
        return debits

    cat_median = debits.groupby("category")["amount"].transform("median")
    features = pd.DataFrame({
        "log_amount": np.log1p(debits["amount"]),
        "hour": debits["hour"],
        "amount_vs_category": debits["amount"] / cat_median.replace(0, 1),
        "is_night": debits["hour"].between(0, 5).astype(int),
    })

    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    model.fit(features)
    debits["is_anomaly"] = model.predict(features) == -1
    debits["anomaly_score"] = -model.score_samples(features)   # higher = more unusual
    return debits


def evaluate(debits: pd.DataFrame):
    """Works only because our synthetic data has a hidden answer column."""
    if "true_anomaly_label" not in debits.columns:
        return None
    truth = debits["true_anomaly_label"] == 1
    pred = debits["is_anomaly"]
    tp = int((truth & pred).sum())
    precision = tp / pred.sum() if pred.sum() else 0.0
    recall = tp / truth.sum() if truth.sum() else 0.0
    return {"true_anomalies": int(truth.sum()), "flagged": int(pred.sum()),
            "caught": tp, "precision": round(float(precision), 2), "recall": round(float(recall), 2)}
