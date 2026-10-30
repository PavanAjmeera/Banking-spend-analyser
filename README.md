# Daily Banking Spend Analyzer

## Problem
Customers struggle to understand where their money goes, notice suspicious payments, or plan next month.
We built a small data-science pipeline and dashboard that answers: *Where did I spend? Is anything unusual? What will I spend next month?*

## Our approach
Data -> Clean -> Categorise -> Detect unusual transactions -> Forecast -> Dashboard

| Step | File | Method | Why |
|---|---|---|---|
| Data | `src/generate_data.py` | Synthetic 12-month bank statement (intentionally messy) | Real data is private |
| Clean + categorise | `src/categorise.py` | Remove duplicates/missing values, keyword rules | Simple, explainable |
| Unusual transactions | `src/anomaly.py` | Isolation Forest | Unsupervised, no fraud labels needed |
| Forecast | `src/forecast.py` | Linear Regression vs 3-month-average baseline, backtested | Honest comparison, MAE in rupees |
| Dashboard | `app/main.py` | Streamlit + Plotly | Fast to build, pure Python |

## How to run
```
pip install -r requirements.txt
python src/generate_data.py
streamlit run app/main.py
```
Optional terminal check: `python src/run_pipeline.py`

## Features (user stories)
1. As a customer, I see my spending by category for any month.
2. As a customer, I see transactions flagged as unusual.
3. As a customer, I see a prediction of next month's spend and how reliable it is.
4. As a customer, I can upload my own CSV and get friendly errors if it is invalid.

## Results (synthetic data)
- Isolation Forest caught all 14 planted anomalies (recall 1.0, precision about 0.88).
- Forecast average error about Rs 1,800 on a 3-month backtest.

## Team and who did what
- Name 1: ...
- Name 2: ...

## Limitations and next steps
- Synthetic data; keyword rules miss unseen merchants (next: text classifier).
- Only 12 months of history; no seasonality (next: seasonal models).
- No authentication or real bank integration.
