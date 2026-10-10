import pandas as pd


# ==========================================
# 🤖 AI FINANCIAL REVIEW
# ==========================================

def generate_financial_review(df):

    if df.empty:
        return "There are no expenses available yet for analysis."

    # Make sure Amount is numeric
    df = df.copy()

    df["Amount"] = pd.to_numeric(
        df["Amount"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Amount"]
    )

    if df.empty:
        return "There is not enough valid expense data for analysis."

    # ==========================================
    # BASIC STATISTICS
    # ==========================================

    total = df["Amount"].sum()
    count = len(df)

    average = (
        total / count
        if count > 0
        else 0
    )

    # ==========================================
    # CATEGORY ANALYSIS
    # ==========================================

    category_totals = (
        df.groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
    )

    if category_totals.empty:
        return "There are no categories available for analysis."

    highest_category = category_totals.index[0]
    highest_amount = category_totals.iloc[0]

    percentage = (
        (highest_amount / total) * 100
        if total > 0
        else 0
    )

    # ==========================================
    # LARGEST TRANSACTION
    # ==========================================

    largest_index = df["Amount"].idxmax()

    largest_transaction = df.loc[
        largest_index
    ]

    largest_amount = largest_transaction["Amount"]

    largest_description = str(
        largest_transaction["Description"]
    )

    # ==========================================
    # AI REVIEW
    # ==========================================

    review = f"""
### 🤖 AI Financial Review

You have recorded **₹{total:,.2f}**
across **{count} transactions**.

Your average transaction is
**₹{average:,.2f}**.

Your highest spending category is
**{highest_category}**, with
**₹{highest_amount:,.2f}**.

This represents approximately
**{percentage:.1f}%** of your total spending.

Your largest individual transaction was
**{largest_description}**, costing
**₹{largest_amount:,.2f}**.
"""

    # ==========================================
    # CATEGORY ANALYSIS
    # ==========================================

    if percentage >= 40:

        review += f"""
### ⚠️ High Spending Concentration

A large portion of your spending is concentrated
in **{highest_category}**.

Review your recent {str(highest_category).lower()}
expenses and identify purchases that could be reduced.
"""

    elif percentage >= 25:

        review += f"""
### 💡 Category Insight

**{highest_category}** is your largest spending category.

Keeping this category under control could help
improve your savings.
"""

    else:

        review += """
### ✅ Spending Distribution

Your spending is relatively distributed across
different categories.

Continue monitoring your expenses regularly.
"""

    # ==========================================
    # LARGE TRANSACTION DETECTION
    # ==========================================

    if average > 0 and largest_amount > average * 3:

        review += f"""
### 🚨 Large Transaction Detected

Your **₹{largest_amount:,.2f}** transaction for
**{largest_description}** is significantly higher
than your average transaction.

Consider checking whether this was a necessary
or planned purchase.
"""

    # ==========================================
    # SMALL TRANSACTION ANALYSIS
    # ==========================================

    small_transactions = df[
        df["Amount"] < average * 0.25
    ]

    if len(small_transactions) >= 5:

        review += f"""
### 🔎 Small Expense Pattern

You have **{len(small_transactions)}**
relatively small transactions.

Small purchases can accumulate over time, so
review recurring low-value expenses as well.
"""

    # ==========================================
    # FINAL RECOMMENDATION
    # ==========================================

    review += """
### 💡 AI Recommendation

Review your highest spending category regularly,
monitor unusually large transactions, and compare
your spending against your monthly budget.

Consistent monitoring can help identify unnecessary
expenses and improve your financial planning.
"""

    return review


# ==========================================
# 💬 FINANCIAL QUESTION ANSWER
# ==========================================

def answer_financial_question(df, question):


    if df.empty:
        return "I don't have enough expense data to answer that yet."

    df = df.copy()

    # ==========================================
    # SUPPORT BOTH STREAMLIT AND CSV COLUMN NAMES
    # ==========================================

    if "Amount" not in df.columns and "amount" in df.columns:
        df["Amount"] = df["amount"]

    if "Category" not in df.columns and "category" in df.columns:
        df["Category"] = df["category"]

    if "Description" not in df.columns and "notes" in df.columns:
        df["Description"] = df["notes"]

    # ==========================================
    # CHECK REQUIRED COLUMNS
    # ==========================================

    required = ["Amount", "Category"]

    for column in required:
        if column not in df.columns:
            return f"I cannot analyze the data because the '{column}' column is missing."

    # ==========================================
    # CLEAN AMOUNT
    # ==========================================

    df["Amount"] = pd.to_numeric(
        df["Amount"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Amount"]
    )

    if df.empty:
        return "I don't have enough valid expense data to answer that."

    total = df["Amount"].sum()

    question = str(question).lower().strip()

    # ==========================================
    # CATEGORY TOTALS
    # ==========================================

    category_totals = (
        df.groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
    )

    if category_totals.empty:
        return "There are no categories available for analysis."

    highest_category = category_totals.index[0]
    highest_amount = category_totals.iloc[0]

    # ==========================================
    # SPECIFIC CATEGORY
    # ==========================================

    selected_category = None

    for category in category_totals.index:

        if str(category).lower() in question:
            selected_category = category
            break

    if selected_category is not None:

        category_amount = category_totals[
            selected_category
        ]

        category_count = len(
            df[
                df["Category"] == selected_category
            ]
        )

        return (
            f"You spent **₹{category_amount:,.2f}** "
            f"on **{selected_category}** across "
            f"**{category_count} transactions**."
        )

    # ==========================================
    # BIGGEST EXPENSE
    # ==========================================

    if (
        "biggest expense" in question
        or "largest expense" in question
        or "biggest transaction" in question
        or "largest transaction" in question
    ):

        largest = df.loc[
            df["Amount"].idxmax()
        ]

        description = largest.get(
            "Description",
            "Unknown transaction"
        )

        category = largest.get(
            "Category",
            "Unknown"
        )

        return (
            f"Your biggest transaction was "
            f"**{description}**, costing "
            f"**₹{largest['Amount']:,.2f}**, "
            f"from the **{category}** category."
        )

    # ==========================================
    # TOTAL SPENDING
    # ==========================================

    if (
        "total spending" in question
        or "total spent" in question
        or "how much have i spent" in question
        or "how much did i spend" in question
    ):

        return (
            f"Your total recorded spending is "
            f"**₹{total:,.2f}** across "
            f"**{len(df)} transactions**."
        )

    # ==========================================
    # HIGHEST CATEGORY
    # ==========================================

    if (
        "most" in question
        or "highest" in question
        or "largest category" in question
        or "spend the most" in question
    ):

        return (
            f"Your highest spending category is "
            f"**{highest_category}** with "
            f"**₹{highest_amount:,.2f}** spent."
        )

    # ==========================================
    # REDUCE / SAVE
    # ==========================================

    if (
        "reduce" in question
        or "save money" in question
        or "save more" in question
        or "saving" in question
    ):

        return (
            f"Start by reviewing your "
            f"**{highest_category}** expenses. "
            f"You have spent **₹{highest_amount:,.2f}** "
            f"in this category. Reducing unnecessary "
            f"expenses here could have the biggest impact."
        )

    # ==========================================
    # BUDGET
    # ==========================================

    if "budget" in question:

        return (
            f"Your recorded spending is "
            f"**₹{total:,.2f}**. Compare this with "
            f"your monthly budget to determine whether "
            f"you are staying within your planned limit."
        )

    # ==========================================
    # SPENDING HABITS
    # ==========================================

    if (
        "habit" in question
        or "spending pattern" in question
        or "spending behavior" in question
    ):

        return (
            f"Your largest spending category is "
            f"**{highest_category}** at "
            f"**₹{highest_amount:,.2f}**. "
            f"Review recurring expenses in this category "
            f"and look for unnecessary purchases."
        )

    # ==========================================
    # DEFAULT
    # ==========================================

    return (
        f"You have spent **₹{total:,.2f}** so far. "
        f"Your largest spending category is "
        f"**{highest_category}**."
    )