import pandas as pd
import sqlite3

from modules.lstm_forecasting import forecast_with_lstm


# Connect to FinSight database
conn = sqlite3.connect("database/finsight.db")

df = pd.read_sql(
    "SELECT * FROM expenses",
    conn
)

conn.close()


# Run LSTM forecast
result = forecast_with_lstm(df)


print("\n===== LSTM TEST =====")

print("Success:", result["success"])


if result["success"]:

    print(
        "Next Month:",
        result["next_month"]
    )

    print(
        f"Forecast: ₹{result['forecast']:,.2f}"
    )

    print(
        "Months Used:",
        result["months_used"]
    )

else:

    print(
        "Message:",
        result["message"]
    )