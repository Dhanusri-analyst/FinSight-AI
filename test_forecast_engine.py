import sqlite3
import pandas as pd

from modules.forecast_engine import forecast_spending


conn = sqlite3.connect("database/finsight.db")

df = pd.read_sql(
    "SELECT * FROM expenses",
    conn
)

conn.close()


result = forecast_spending(df)


print("\n================================")
print("FinSight AI Forecast Engine")
print("================================")

print("Success:", result["success"])
print("Model:", result["model"])

if result["success"]:

    print(
        f"Prediction: ₹{result['prediction']:,.2f}"
    )

    if "months_available" in result:

        print(
            "Months available:",
            result["months_available"]
        )

else:

    print(
        "Message:",
        result["message"]
    )