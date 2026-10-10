from modules.database import *
from modules.categorizer import predict_category

create_table()

while True:
    print("\n==============================")
    print("        FinSight AI")
    print("==============================")
    print("1. Add Expense")
    print("2. View Expenses")
    print("3. Total Spending")
    print("4. Exit")

    choice = input("\nEnter your choice: ")

    if choice == "1":

        description = input(
            "Enter expense description (e.g. Uber ride, grocery shopping): "
        )

        try:
            amount = float(input("Enter amount (₹): "))

            if amount <= 0:
                print("\n❌ Amount must be greater than 0.")
                continue

        except ValueError:
            print("\n❌ Please enter a valid amount.")
            continue

        # ==========================================
        # AI CATEGORY PREDICTION
        # ==========================================

        category = predict_category(description)

        print(f"\n🤖 AI Predicted Category: {category}")

        # ==========================================
        # SAVE TO DATABASE
        # ==========================================

        add_expense(description, amount, category)

        print("✅ Expense Added Successfully!")

    elif choice == "2":

        expenses = get_expenses()

        if len(expenses) == 0:
            print("\nNo expenses found.")

        else:
            print("\n========== Expense History ==========\n")

            for expense in expenses:
                print(f"ID          : {expense[0]}")
                print(f"Description : {expense[1]}")
                print(f"Amount      : ₹{expense[2]}")
                print(f"Category    : {expense[3]}")
                print(f"Date        : {expense[4]}")
                print("------------------------------------")

    elif choice == "3":

        print(f"\n💰 Total Spending : ₹{total_spending()}")

    elif choice == "4":

        print("\n👋 Thank you for using FinSight AI")
        break

    else:
        print("\n❌ Invalid Choice")
