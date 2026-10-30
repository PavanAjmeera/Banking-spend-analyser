"""
STEP 5 - DASHBOARD (Streamlit)
Run from the project folder:   streamlit run app/main.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # so we can import from src/

import pandas as pd
import plotly.express as px
import streamlit as st

from src.anomaly import detect_anomalies, evaluate
from src.categorise import load_and_prepare
from src.forecast import category_forecast, forecast_next_month, monthly_spend

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "transactions.csv"

st.set_page_config(page_title="Daily Banking Spend Dashboard", layout="wide")
st.title("Daily Banking Spend Dashboard")
st.caption("Categorise transactions | Monthly spend | Unusual transactions | Next-month forecast")

# ---------- sidebar ----------
st.sidebar.header("Settings")
uploaded = st.sidebar.file_uploader("Upload your own transactions CSV (optional)", type="csv")
contamination = st.sidebar.slider(
    "Unusual-transaction sensitivity", 0.01, 0.10, 0.03, 0.01,
    help="Roughly what share of transactions we expect to be unusual. Higher = flags more.")

# ---------- load data (with friendly error handling = edge cases) ----------
try:
    df, log = load_and_prepare(uploaded if uploaded else DATA_PATH)
except FileNotFoundError:
    st.error("No data file found. In the terminal run:  python src/generate_data.py")
    st.stop()
except ValueError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

debits = detect_anomalies(df, contamination)
flagged_ids = set(debits[debits["is_anomaly"]]["txn_id"])
monthly = monthly_spend(df, exclude_ids=flagged_ids)

tab1, tab2, tab3, tab4 = st.tabs(["Monthly spend", "Unusual transactions", "Next-month forecast", "Data and cleaning"])

# ---------- TAB 1: monthly spend ----------
with tab1:
    months = sorted(df["month"].unique())
    month = st.selectbox("Month", months, index=len(months) - 1)
    m_df = df[df["month"] == month]
    m_debit = m_df[m_df["txn_type"] == "DEBIT"]
    income = m_df[m_df["txn_type"] == "CREDIT"]["amount"].sum()
    spent = m_debit["amount"].sum()

    prev_idx = months.index(month) - 1
    delta = None
    if prev_idx >= 0:
        prev = df[(df["month"] == months[prev_idx]) & (df["txn_type"] == "DEBIT")]["amount"].sum()
        if prev > 0:
            delta = f"{(spent - prev) / prev * 100:+.1f}% vs last month"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total spent", f"₹{spent:,.0f}", delta, delta_color="inverse")
    c2.metric("Income", f"₹{income:,.0f}")
    c3.metric("Saved", f"₹{income - spent:,.0f}")
    c4.metric("Transactions", len(m_debit))

    if m_debit.empty:
        st.info("No spending found for this month.")
    else:
        left, right = st.columns(2)
        by_cat = m_debit.groupby("category")["amount"].sum().reset_index()
        left.plotly_chart(px.pie(by_cat, names="category", values="amount", title="Spend by category"),
                          width="stretch")
        trend = df[df["txn_type"] == "DEBIT"].groupby("month")["amount"].sum().reset_index()
        right.plotly_chart(px.bar(trend, x="month", y="amount", title="Monthly spend trend (all debits)"),
                           width="stretch")
        st.subheader("Biggest transactions this month")
        st.dataframe(m_debit.nlargest(5, "amount")[["timestamp", "description", "category", "amount"]],
                     width="stretch", hide_index=True)

# ---------- TAB 2: unusual transactions ----------
with tab2:
    st.write("**Isolation Forest** flags transactions that look different from normal behaviour: "
             "very large amounts, odd hours, or far above that category's usual amount.")
    flagged = debits[debits["is_anomaly"]].sort_values("anomaly_score", ascending=False)
    st.metric("Flagged transactions", len(flagged))
    if flagged.empty:
        st.success("No unusual transactions found.")
    else:
        st.dataframe(flagged[["txn_id", "timestamp", "description", "category", "amount", "anomaly_score"]],
                     width="stretch", hide_index=True)
    fig = px.scatter(debits, x="hour", y="amount", color="is_anomaly", log_y=True,
                     hover_data=["description"], title="Amount vs hour of day (log scale)")
    st.plotly_chart(fig, width="stretch")

    result = evaluate(debits)
    if result:
        with st.expander("How accurate is it? (possible because our data is synthetic)"):
            st.write(result)
            st.caption("Precision = of the flagged, how many were truly unusual. "
                       "Recall = of the truly unusual, how many we caught. "
                       "For fraud, recall matters most: missing fraud is costlier than a false alarm.")

# ---------- TAB 3: forecast ----------
with tab3:
    fc = forecast_next_month(monthly)
    if fc is None:
        st.warning("Need at least 5 months of data to forecast.")
    else:
        st.metric(f"Predicted spend for {fc['next_month']}", f"₹{fc['prediction']:,.0f}",
                  help=f"Best model on backtest: {fc['best_model']}")
        chart = monthly.copy()
        chart["type"] = "Actual (normal spend)"
        chart = pd.concat([chart, pd.DataFrame([{"month": fc["next_month"], "spend": fc["prediction"],
                                                 "type": "Forecast"}])])
        st.plotly_chart(px.bar(chart, x="month", y="spend", color="type", title="Actual vs forecast"),
                        width="stretch")
        st.subheader("Model comparison (backtest on last 3 months)")
        st.table(pd.DataFrame({
            "Model": ["Linear Regression (trend)", "Average of last 3 months (baseline)"],
            "Avg error (₹)": [round(fc["mae_linear"]), round(fc["mae_baseline"])]}))
        st.caption("Lower error is better. We forecast on normal spending only, "
                   "after removing flagged unusual transactions.")
        st.subheader("Expected spend by category next month")
        st.dataframe(category_forecast(df, exclude_ids=flagged_ids), width="stretch", hide_index=True)

# ---------- TAB 4: data ----------
with tab4:
    st.subheader("Cleaning log")
    st.json(log)
    cats = ["All"] + sorted(df["category"].unique())
    pick = st.selectbox("Filter by category", cats)
    view = df if pick == "All" else df[df["category"] == pick]
    st.dataframe(view[["txn_id", "timestamp", "description", "category", "amount", "txn_type"]],
                 width="stretch", hide_index=True)
    st.download_button("Download categorised data (CSV)", df.to_csv(index=False), "categorised.csv")
