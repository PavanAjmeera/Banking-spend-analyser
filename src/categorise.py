"""
STEP 2 - CLEANING + CATEGORISING
Loads the CSV, cleans it, and gives every transaction a category.

Approach: simple keyword rules. Why? They are fast to build, 100% explainable,
and easy to extend. Trade-off: unseen merchants fall into "Other".
(Next step with more time: train a text classifier, e.g. TF-IDF + Logistic Regression.)
"""
import pandas as pd

REQUIRED_COLUMNS = ["txn_id", "timestamp", "description", "amount", "txn_type"]

# first matching rule wins, so order matters
CATEGORY_KEYWORDS = {
    "Income": ["SALARY"],
    "EMI/Loans": ["EMI", "LOAN"],
    "Rent": ["RENT"],
    "Food": ["SWIGGY", "ZOMATO", "GROCERY", "STARBUCKS", "BAZAAR"],
    "Transport": ["UBER", "OLA", "FUEL", "METRO"],
    "Shopping": ["AMAZON", "FLIPKART", "MYNTRA"],
    "Bills": ["AIRTEL", "TNEB", "ELECTRICITY", "JIO"],
    "Entertainment": ["NETFLIX", "BOOKMYSHOW"],
    "Health": ["PHARMACY", "APOLLO"],
}


def categorise(description: str) -> str:
    words = description.replace("/", " ").replace("-", " ").split()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(k in words for k in keywords):
            return category
    return "Other"


def clean_transactions(df: pd.DataFrame):
    """Returns (clean_df, log). The log is great to show the panel."""
    log = {"rows_loaded": len(df)}

    before = len(df)
    df = df.drop_duplicates(subset="txn_id").copy()
    log["duplicates_removed"] = before - len(df)

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["amount"])
    log["missing_or_invalid_amounts_removed"] = before - len(df)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["timestamp"])
    log["bad_dates_removed"] = before - len(df)

    df["description"] = df["description"].fillna("").astype(str).str.strip().str.upper()
    df["txn_type"] = df["txn_type"].astype(str).str.strip().str.upper()

    log["rows_after_cleaning"] = len(df)
    return df.reset_index(drop=True), log


def load_and_prepare(source):
    """source = file path or an uploaded file. Raises ValueError with a friendly message."""
    df = pd.read_csv(source)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Your file is missing these columns: {missing}. Needed: {REQUIRED_COLUMNS}")
    if df.empty:
        raise ValueError("The file has no rows.")

    df, log = clean_transactions(df)
    if df.empty:
        raise ValueError("No valid rows left after cleaning.")

    df["category"] = df["description"].apply(categorise)
    df["date"] = df["timestamp"].dt.date
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    log["categories_found"] = int(df["category"].nunique())
    log["uncategorised_rows"] = int((df["category"] == "Other").sum())
    return df, log
