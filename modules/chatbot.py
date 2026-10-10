import pandas as pd
from modules.anomaly_detector import detect_spending_anomalies


def answer_option(option, df, budget=0):
    """Answer predefined finance questions using expense data."""

    if df is None or df.empty:
        return "No expenses available yet."

    df = df.copy()

    required = {"Amount", "Category"}
    if not required.issubset(df.columns):
        return "Expense data must contain Amount and Category columns."

    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df = df.dropna(subset=["Amount"])

    if df.empty:
        return "No valid expense amounts are available."

    if option == "💰 Total Spending":
        total = df["Amount"].sum()
        return f"💰 Total spending: ₹{total:,.2f}"

    elif option == "💵 Remaining Budget":
        if budget <= 0:
            return "🎯 Please set your monthly budget first."

        remaining = budget - df["Amount"].sum()

        if remaining < 0:
            return f"⚠️ You are over budget by ₹{abs(remaining):,.2f}."

        return f"💵 Remaining Budget: ₹{remaining:,.2f}"

    elif option == "🍔 Food Expenses":
        amount = df.loc[df["Category"] == "Food", "Amount"].sum()
        return f"🍔 Food expenses: ₹{amount:,.2f}"

    elif option == "🎓 Education Expenses":
        amount = df.loc[df["Category"] == "Education", "Amount"].sum()
        return f"🎓 Education expenses: ₹{amount:,.2f}"

    elif option == "🛒 Shopping Expenses":
        amount = df.loc[df["Category"] == "Shopping", "Amount"].sum()
        return f"🛒 Shopping expenses: ₹{amount:,.2f}"

    elif option == "🚗 Transport Expenses":
        amount = df.loc[df["Category"] == "Transport", "Amount"].sum()
        return f"🚗 Transport expenses: ₹{amount:,.2f}"

    elif option == "📊 Highest Spending Category":
        totals = df.groupby("Category")["Amount"].sum()

        if totals.empty:
            return "No categorized expenses are available."

        return (
            f"📊 Highest spending category: {totals.idxmax()} "
            f"(₹{totals.max():,.2f})"
        )

    elif option == "🤖 AI Recommendation":
        totals = df.groupby("Category")["Amount"].sum()

        if totals.empty:
            return "No categorized expenses are available."

        category = totals.idxmax()
        amount = totals.max()

        return (
            f"🤖 You spend the most on {category} "
            f"(₹{amount:,.2f}). Review transactions in this "
            "category to identify possible savings."
        )

    elif option == "🚨 Unusual Expenses":
        valid_count = int((df["Amount"] > 0).sum())

        if valid_count < 5:
            return (
                "🚨 Not enough transactions to detect unusual expenses. "
                f"You have {valid_count} valid transactions; at least "
                "5 positive-amount transactions are required."
            )

        # Run the existing Isolation Forest detector.
        try:
            results = detect_spending_anomalies(df)
        except Exception as error:
            return (
                "⚠️ Anomaly detection could not be completed. "
                f"Please check the detector configuration. Details: {error}"
            )

        unusual = results[results["Is_Anomaly"] == True].copy()

        if unusual.empty:
            return (
                "✅ No unusual transactions were flagged by the "
                "Isolation Forest model. This does not guarantee "
                "that every transaction is normal."
            )

        # Show the most unusual transactions first.
        unusual = unusual.sort_values(
            "Anomaly_Score", ascending=True
        )

        lines = [
            f"🚨 Isolation Forest flagged {len(unusual)} unusual "
            "transaction(s):",
            ""
        ]

        for _, row in unusual.head(10).iterrows():
            description = row.get("Description", "Expense")
            category = row.get("Category", "Unknown")
            amount = row["Amount"]

            details = (
                f"• {description} — ₹{amount:,.2f} "
                f"({category})"
            )

            if "Date" in results.columns and pd.notna(row.get("Date")):
                details += f" | Date: {row['Date']}"

            lines.append(details)

        if len(unusual) > 10:
            lines.append(
                f"\n...and {len(unusual) - 10} more flagged transactions."
            )

        lines.append(
            "\n💡 Review these transactions to check whether they "
            "are genuine. An anomaly is a statistical flag, not proof "
            "of fraud or an incorrect expense."
        )

        return "\n".join(lines)

    return (
        "I don't recognize that question yet. "
        "Please choose one of the available finance questions."
    )
      
