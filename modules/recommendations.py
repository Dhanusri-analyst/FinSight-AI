# ============================================================
# FINSIGHT AI - PERSONALIZED RECOMMENDATION ENGINE
# ============================================================

import pandas as pd


def generate_recommendations(
    df,
    budget=0,
    forecast=None,
    anomaly_count=0
):
    """
    Generate personalized financial recommendations
    using spending, budget, anomaly and forecast information.
    """

    recommendations = []

    if df is None or df.empty:
        return [
            {
                "type": "info",
                "title": "Getting Started",
                "message": (
                    "Add more transactions so FinSight AI "
                    "can generate personalized recommendations."
                )
            }
        ]

    # --------------------------------------------------------
    # DATA PREPARATION
    # --------------------------------------------------------

    data = df.copy()

    data["Amount"] = pd.to_numeric(
        data["Amount"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["Amount"]
    )

    if data.empty:
        return [
            {
                "type": "info",
                "title": "Insufficient Data",
                "message": (
                    "Valid transaction amounts are required "
                    "to generate recommendations."
                )
            }
        ]

    # --------------------------------------------------------
    # BASIC SPENDING METRICS
    # --------------------------------------------------------

    total_spending = data["Amount"].sum()

    average_transaction = data["Amount"].mean()

    highest_transaction = data["Amount"].max()

    highest_transaction_row = data.loc[
        data["Amount"].idxmax()
    ]

    highest_category = None
    highest_category_amount = 0

    if "Category" in data.columns:

        category_summary = (
            data.groupby("Category")["Amount"]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        if not category_summary.empty:

            highest_category = (
                category_summary.index[0]
            )

            highest_category_amount = (
                category_summary.iloc[0]
            )

    # --------------------------------------------------------
    # 1. BUDGET RECOMMENDATION
    # --------------------------------------------------------

    if budget and budget > 0:

        budget_percentage = (
            total_spending / budget
        ) * 100

        if budget_percentage > 100:

            recommendations.append(
                {
                    "type": "danger",
                    "title": "Budget Exceeded",
                    "message": (
                        f"You have exceeded your "
                        f"₹{budget:,.2f} budget by "
                        f"₹{total_spending - budget:,.2f}. "
                        "Review your highest discretionary "
                        "expenses and reduce non-essential "
                        "spending."
                    )
                }
            )

        elif budget_percentage >= 90:

            recommendations.append(
                {
                    "type": "warning",
                    "title": "Budget Almost Used",
                    "message": (
                        f"You have used "
                        f"{budget_percentage:.1f}% of your "
                        "monthly budget. Consider limiting "
                        "non-essential spending."
                    )
                }
            )

        elif budget_percentage >= 75:

            recommendations.append(
                {
                    "type": "warning",
                    "title": "Watch Your Budget",
                    "message": (
                        f"You have already used "
                        f"{budget_percentage:.1f}% of your "
                        "monthly budget. Keep an eye on "
                        "your remaining expenses."
                    )
                }
            )

        else:

            recommendations.append(
                {
                    "type": "success",
                    "title": "Budget On Track",
                    "message": (
                        f"You have used "
                        f"{budget_percentage:.1f}% of your "
                        "monthly budget. Your current "
                        "spending is within a comfortable range."
                    )
                }
            )

    # --------------------------------------------------------
    # 2. HIGHEST CATEGORY RECOMMENDATION
    # --------------------------------------------------------

    if highest_category is not None and total_spending > 0:

        category_percentage = (
            highest_category_amount /
            total_spending
        ) * 100

        if category_percentage >= 50:

            recommendations.append(
                {
                    "type": "warning",
                    "title": "High Category Concentration",
                    "message": (
                        f"{highest_category} accounts for "
                        f"{category_percentage:.1f}% of your "
                        "total spending. Review this category "
                        "carefully and separate essential "
                        "expenses from discretionary ones."
                    )
                }
            )

        elif category_percentage >= 30:

            recommendations.append(
                {
                    "type": "info",
                    "title": "Largest Spending Category",
                    "message": (
                        f"{highest_category} is your largest "
                        f"spending category at "
                        f"{category_percentage:.1f}% of total "
                        "spending. Monitor this category "
                        "regularly."
                    )
                }
            )

    # --------------------------------------------------------
    # 3. ANOMALY RECOMMENDATION
    # --------------------------------------------------------

    if anomaly_count and anomaly_count > 0:

        recommendations.append(
            {
                "type": "danger",
                "title": "Unusual Spending Detected",
                "message": (
                    f"FinSight AI detected "
                    f"{anomaly_count} unusual transaction(s). "
                    "Review these transactions to make sure "
                    "they were intentional."
                )
            }
        )

    # --------------------------------------------------------
    # 4. LARGE TRANSACTION RECOMMENDATION
    # --------------------------------------------------------

    if (
        average_transaction > 0
        and highest_transaction
        >= average_transaction * 3
    ):

        description = str(
            highest_transaction_row.get(
                "Description",
                "transaction"
            )
        )

        recommendations.append(
            {
                "type": "warning",
                "title": "Large Transaction",
                "message": (
                    f"Your ₹{highest_transaction:,.2f} "
                    f"transaction ({description}) is more "
                    "than three times your average transaction. "
                    "Consider reviewing this expense."
                )
            }
        )

    # --------------------------------------------------------
    # 5. FORECAST RECOMMENDATION
    # --------------------------------------------------------

    if (
        forecast is not None
        and budget
        and budget > 0
    ):

        try:

            forecast_value = float(
                forecast
            )

            if forecast_value > budget:

                recommendations.append(
                    {
                        "type": "danger",
                        "title": "Forecast Above Budget",
                        "message": (
                            f"Your projected spending of "
                            f"₹{forecast_value:,.2f} is above "
                            f"your ₹{budget:,.2f} budget. "
                            "Consider reducing discretionary "
                            "expenses."
                        )
                    }
                )

            elif forecast_value >= budget * 0.9:

                recommendations.append(
                    {
                        "type": "warning",
                        "title": "Forecast Near Budget",
                        "message": (
                            f"Your projected spending of "
                            f"₹{forecast_value:,.2f} is close "
                            "to your monthly budget. "
                            "Monitor upcoming expenses."
                        )
                    }
                )

        except (
            ValueError,
            TypeError
        ):
            pass

    # --------------------------------------------------------
    # 6. SAVINGS RECOMMENDATION
    # --------------------------------------------------------

    if budget and budget > 0:

        remaining_budget = (
            budget - total_spending
        )

        if (
            remaining_budget > 0
            and total_spending < budget * 0.70
        ):

            recommendations.append(
                {
                    "type": "success",
                    "title": "Savings Opportunity",
                    "message": (
                        f"You currently have "
                        f"₹{remaining_budget:,.2f} "
                        "remaining in your budget. "
                        "Consider allocating part of it "
                        "toward savings or investments."
                    )
                }
            )

    # --------------------------------------------------------
    # 7. GENERAL RECOMMENDATION
    # --------------------------------------------------------

    if not recommendations:

        recommendations.append(
            {
                "type": "info",
                "title": "Keep Monitoring",
                "message": (
                    "Your current spending pattern does not "
                    "trigger any major financial warnings. "
                    "Continue monitoring your expenses regularly."
                )
            }
        )

    # --------------------------------------------------------
    # LIMIT OUTPUT
    # --------------------------------------------------------

    return recommendations[:6]