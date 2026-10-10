# ============================================================
# FinSight AI — Production LSTM Spending Forecasting
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/spending_lstm_production.keras"
SCALER_PATH = "models/spending_scaler.pkl"

LOOKBACK = 6


# ============================================================
# LSTM FORECAST FUNCTION
# ============================================================

def forecast_with_lstm(df):
    """
    Predict next month's spending using the production-trained
    LSTM model and its saved scaler.

    Requires at least 6 months of historical spending data.
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if df is None or df.empty:
        return {
            "success": False,
            "message": "No spending data available."
        }

    required_columns = {"Date", "Amount"}

    if not required_columns.issubset(df.columns):
        return {
            "success": False,
            "message": "Date and Amount columns are required."
        }

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    data = df.copy()

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    data["Amount"] = pd.to_numeric(
        data["Amount"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["Date", "Amount"]
    )

    # Prevent negative spending values
    data = data[data["Amount"] >= 0]

    if data.empty:
        return {
            "success": False,
            "message": "No valid spending data available."
        }

    # --------------------------------------------------------
    # Aggregate spending by month
    # --------------------------------------------------------

    monthly = (
        data
        .set_index("Date")
        .resample("MS")["Amount"]
        .sum()
        .reset_index()
    )

    monthly = monthly.sort_values("Date")

    # --------------------------------------------------------
    # LSTM requires 6 months of history
    # --------------------------------------------------------

    if len(monthly) < LOOKBACK:
        return {
            "success": False,
            "message": (
                f"LSTM requires at least "
                f"{LOOKBACK} months of spending history. "
                f"Currently available: {len(monthly)} months."
            )
        }

    # --------------------------------------------------------
    # Check production model
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):
        return {
            "success": False,
            "message": (
                "Production LSTM model is not available yet."
            )
        }

    # --------------------------------------------------------
    # Check saved scaler
    # --------------------------------------------------------

    if not os.path.exists(SCALER_PATH):
        return {
            "success": False,
            "message": (
                "Production LSTM scaler is not available yet."
            )
        }

    try:

        # ----------------------------------------------------
        # Load production-trained model
        # ----------------------------------------------------

        model = tf.keras.models.load_model(
            MODEL_PATH
        )

        # ----------------------------------------------------
        # Load production scaler
        # ----------------------------------------------------

        scaler = joblib.load(
            SCALER_PATH
        )

        # ----------------------------------------------------
        # Prepare monthly spending values
        # ----------------------------------------------------

        spending = (
            monthly["Amount"]
            .astype(float)
            .values
            .reshape(-1, 1)
        )

        # ----------------------------------------------------
        # Transform using SAVED scaler
        # ----------------------------------------------------

        scaled_data = scaler.transform(
            spending
        )

        # ----------------------------------------------------
        # Take latest 6 months
        # ----------------------------------------------------

        last_sequence = scaled_data[-LOOKBACK:]

        X_input = last_sequence.reshape(
            1,
            LOOKBACK,
            1
        )

        # ----------------------------------------------------
        # Predict next month
        # ----------------------------------------------------

        prediction_scaled = model(
            X_input,
            training=False
        ).numpy()

        prediction = scaler.inverse_transform(
            prediction_scaled
        )[0][0]

        prediction = max(
            0.0,
            float(prediction)
        )

        # ----------------------------------------------------
        # Determine next month
        # ----------------------------------------------------

        last_month = monthly["Date"].iloc[-1]

        next_month = (
            last_month
            + pd.DateOffset(months=1)
        )

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return {
            "success": True,
            "forecast": prediction,
            "next_month": next_month,
            "months_used": LOOKBACK,
            "monthly_data": monthly
        }

    except Exception as e:

        return {
            "success": False,
            "message": (
                f"Production LSTM prediction failed: {str(e)}"
            )
        }