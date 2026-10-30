"""
STEP 1 - DATA
Creates a fake bank statement (12 months) and saves it to data/transactions.csv

We use synthetic (fake) data because real customer data is private.
The data is deliberately a bit "dirty" (duplicates, missing amounts, messy text)
so that our cleaning step has real work to do.

Run:  python src/generate_data.py
"""
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(42)      # same seed = same data every time (repeatable)
np.random.seed(42)

START = datetime(2025, 11, 1)
END = datetime(2026, 10, 31)

# (merchant, min amount, max amount, how often)
SPENDING = [
    ("SWIGGY", 150, 700, 10), ("ZOMATO", 150, 700, 8),
    ("BIG BAZAAR GROCERY", 500, 2500, 5), ("STARBUCKS", 200, 600, 4),
    ("UBER", 100, 600, 8), ("OLA CABS", 100, 500, 5),
    ("INDIAN OIL FUEL", 500, 3000, 3), ("METRO RECHARGE", 200, 1000, 2),
    ("AMAZON", 300, 5000, 6), ("FLIPKART", 300, 5000, 4), ("MYNTRA", 500, 4000, 3),
    ("BOOKMYSHOW", 250, 1200, 2), ("APOLLO PHARMACY", 100, 1500, 2),
]
MERCHANTS = [m[0] for m in SPENDING]
WEIGHTS = [m[3] for m in SPENDING]
RANGES = {m[0]: (m[1], m[2]) for m in SPENDING}

rows = []


def add(ts, desc, amount, txn_type, label=0):
    rows.append({"timestamp": ts, "description": desc, "amount": round(amount, 2),
                 "txn_type": txn_type, "true_anomaly_label": label})


def ref():
    return random.randint(100000, 999999)


day = START
while day <= END:
    # random everyday spending
    for _ in range(random.choices([0, 1, 2, 3], [0.25, 0.4, 0.25, 0.1])[0]):
        merchant = random.choices(MERCHANTS, WEIGHTS)[0]
        lo, hi = RANGES[merchant]
        ts = day.replace(hour=random.randint(8, 22), minute=random.randint(0, 59))
        mode = random.choice(["UPI", "CARD"])
        add(ts, f"{mode}/{merchant}/{ref()}", random.uniform(lo, hi), "DEBIT")

    # fixed monthly items
    d = day.day
    at9 = day.replace(hour=9, minute=0)
    if d == 1:
        add(at9, f"NEFT/ACME TECH SALARY/{ref()}", 62000, "CREDIT")
    if d == 3:
        add(at9, f"NETBANKING/HDFC HOME LOAN EMI/{ref()}", 8500, "DEBIT")
    if d == 5:
        add(at9, f"NETBANKING/RENT TRANSFER/{ref()}", 15000, "DEBIT")
    if d == 8:
        add(at9, f"UPI/AIRTEL POSTPAID/{ref()}", 999, "DEBIT")
    if d == 10:
        add(at9, f"UPI/TNEB ELECTRICITY/{ref()}", random.uniform(800, 2200), "DEBIT")
    if d == 12:
        add(at9, f"UPI/JIO FIBER/{ref()}", 1178, "DEBIT")
    if d == 15:
        add(at9, f"CARD/NETFLIX/{ref()}", 649, "DEBIT")
    day += timedelta(days=1)

# suspicious transactions: very large, at night, unknown merchant
total_days = (END - START).days
for _ in range(10):
    ts = (START + timedelta(days=random.randint(0, total_days))).replace(
        hour=random.randint(1, 4), minute=random.randint(0, 59))
    add(ts, f"CARD/UNKNOWN INTL MERCHANT/{ref()}", random.uniform(40000, 95000), "DEBIT", 1)
for _ in range(4):
    ts = (START + timedelta(days=random.randint(0, total_days))).replace(
        hour=random.randint(1, 4), minute=random.randint(0, 59))
    add(ts, f"UPI/UNKNOWN WALLET TOPUP/{ref()}", random.uniform(15000, 30000), "DEBIT", 1)

df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
df.insert(0, "txn_id", [f"TXN{i:06d}" for i in range(1, len(df) + 1)])

# ---- make the data messy on purpose ----
dupes = df.sample(6, random_state=1)
df = pd.concat([df, dupes]).sort_values("txn_id").reset_index(drop=True)      # 6 duplicate rows
df.loc[df.sample(4, random_state=2).index, "amount"] = np.nan                  # 4 missing amounts
messy = df.sample(frac=0.05, random_state=3).index
df.loc[messy, "description"] = df.loc[messy, "description"].str.lower().apply(lambda s: f"  {s} ")

out = Path(__file__).resolve().parent.parent / "data" / "transactions.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows to {out}")
