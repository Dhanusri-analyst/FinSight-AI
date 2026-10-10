import pandas as pd

INPUT_FILE = "data/budgetwise_finance_dataset.csv"
OUTPUT_FILE = "data/cleaned_budgetwise.csv"

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)

# Remove rows without a category
df = df.dropna(subset=["category"]).copy()

# Standardize category names
df["category"] = df["category"].astype(str).str.strip().str.lower()

category_mapping = {
    "food": "Food",
    "foods": "Food",
    "fod": "Food",
    "foodd": "Food",

    "rent": "Rent",
    "rentt": "Rent",
    "rnt": "Rent",

    "travel": "Travel",
    "traval": "Travel",
    "travl": "Travel",

    "entertain": "Entertainment",
    "entrtnmnt": "Entertainment",
    "entertainment": "Entertainment",

    "utility": "Utilities",
    "utilties": "Utilities",
    "utilities": "Utilities",
    "utlities": "Utilities",

    "education": "Education",
    "educaton": "Education",
    "edu": "Education",

    "health": "Health",
    "helth": "Health",

    "saving": "Savings",
    "savings": "Savings",

    "other": "Others",
    "others": "Others",
    "misc": "Others",

    "salary": "Salary",
    "freelance": "Freelance",
    "investment": "Investment",
    "bonus": "Bonus"
}

df["category"] = df["category"].map(category_mapping)

# Remove any unmapped categories
df = df.dropna(subset=["category"]).copy()

# Clean amount
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df["amount"] = df["amount"].fillna(df["amount"].median())

# Clean text columns
for column in ["notes", "payment_mode", "transaction_type", "location"]:
    df[column] = df[column].fillna("Unknown").astype(str).str.strip()

# Remove duplicate transaction IDs
df = df.drop_duplicates(subset=["transaction_id"])

# Save cleaned dataset
df.to_csv(OUTPUT_FILE, index=False)

print("\n================================")
print("DATA CLEANING COMPLETED")
print("================================")
print("Final shape:", df.shape)

print("\nCategories:")
print(df["category"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

print("\nSaved to:", OUTPUT_FILE)
