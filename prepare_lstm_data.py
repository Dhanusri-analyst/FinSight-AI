import sqlite3
import pandas as pd
import os

# ==========================================
# FinSight AI - Prepare LSTM Training Data
# ==========================================

DB_PATH = "database/finsight.db"
OUTPUT_PATH = "data/monthly_spending.csv"

# Connect to SQLite
conn = sqlite3.connect(DB_PATH)

# Load transactions
df = pd.read_sql_query(
    """
    SELECT date, amount
    FROM expenses
    """,
    conn
)

conn.close()

# ------------------------------------------
# Validate data
# ------------------------------------------

if df.empty:
    print("❌ No transactions found in the database.")
    exit()

# Convert columns
df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df["amount"] = pd.to_numeric(
    df["amount"],
    errors="coerce"
)

# Remove invalid records
df = df.dropna(
    subset=["date", "amount"]
)

df = df[df["amount"] >= 0]

if df.empty:
    print("❌ No valid transactions available.")
    exit()

# ------------------------------------------
# Aggregate transactions by month
# ------------------------------------------

monthly = (
    df.set_index("date")["amount"]
    .resample("MS")
    .sum()
    .reset_index()
)

monthly.columns = [
    "Month",
    "Spending"
]

# ------------------------------------------
# Fill missing months with 0
# ------------------------------------------

full_month_range = pd.date_range(
    start=monthly["Month"].min(),
    end=monthly["Month"].max(),
    freq="MS"
)

monthly = (
    monthly
    .set_index("Month")
    .reindex(
        full_month_range,
        fill_value=0
    )
    .rename_axis("Month")
    .reset_index()
)

# ------------------------------------------
# Save dataset
# ------------------------------------------

os.makedirs(
    "data",
    exist_ok=True
)

monthly.to_csv(
    OUTPUT_PATH,
    index=False
)

# ------------------------------------------
# Display information
# ------------------------------------------

print("\n✅ LSTM training dataset created!")
print(f"📁 Saved to: {OUTPUT_PATH}")

print("\n📊 Dataset preview:")
print(monthly.head(10))

print("\n📊 Dataset information:")
print(f"Months available: {len(monthly)}")
print(f"First month: {monthly['Month'].min().strftime('%Y-%m')}")
print(f"Last month:  {monthly['Month'].max().strftime('%Y-%m')}")

print("\n💰 Spending statistics:")
print(
    monthly["Spending"].describe()
)
