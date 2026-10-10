def answer_option(option, df, budget=0):

    if df.empty:
        return "No expenses available yet."


    if option == "💰 Total Spending":

        total = df["Amount"].sum()

        return f"💰 Total spending: ₹{total:,.2f}"


    elif option == "💵 Remaining Budget":

        spent = df["Amount"].sum()

        remaining = budget - spent

        if budget == 0:
            return "🎯 Please set your monthly budget first."

        return f"💵 Remaining budget: ₹{remaining:,.2f}"


    elif option == "🍔 Food Expenses":

        amount = df[
            df["Category"] == "Food"
        ]["Amount"].sum()

        return f"🍔 Food expenses: ₹{amount:,.2f}"


    elif option == "🎓 Education Expenses":

        amount = df[
            df["Category"] == "Education"
        ]["Amount"].sum()

        return f"🎓 Education expenses: ₹{amount:,.2f}"


    elif option == "🛒 Shopping Expenses":

        amount = df[
            df["Category"] == "Shopping"
        ]["Amount"].sum()

        return f"🛒 Shopping expenses: ₹{amount:,.2f}"


    elif option == "🚗 Transport Expenses":

        amount = df[
            df["Category"] == "Transport"
        ]["Amount"].sum()

        return f"🚗 Transport expenses: ₹{amount:,.2f}"


    elif option == "📊 Highest Spending Category":

        category = (
            df.groupby("Category")["Amount"]
            .sum()
            .idxmax()
        )

        amount = (
            df.groupby("Category")["Amount"]
            .sum()
            .max()
        )

        return f"📊 Highest spending: {category} (₹{amount:,.2f})"


    elif option == "🤖 AI Recommendation":

        category = (
            df.groupby("Category")["Amount"]
            .sum()
            .idxmax()
        )

        return (
            f"🤖 You spend the most on {category}. "
            "Consider monitoring this category to save more."
        )