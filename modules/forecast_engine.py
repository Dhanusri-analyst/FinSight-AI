# ============================================================
# FinSight AI — Automatic Forecast Model Selection
# ============================================================

import pandas as pd

from modules.lstm_forecasting import forecast_with_lstm
from modules.forecasting import forecast_next_month


# ============================================================
# CONFIGURATION
# ============================================================

MIN_STATISTICAL_MONTHS = 4
MIN_LSTM_MONTHS = 6

# IMPORTANT:
# The current LSTM model was trained on synthetic/demo data.
# Therefore it MUST NOT be used for real FinSight predictions.
#
# Change this to True only after:
# 1. Real FinSight data is collected
# 2. LSTM is trained on real data
# 3. The scaler is saved
# 4. The model is validated
# 5. The production model is confirmed ready

PRODUCTION_LSTM_READY = False


# ============================================================
# PREPARE FORECAST DATA
# ============================================================
def prepare_forecast_data(df):
    """
    Normalize database/DataFrame data into:

        Date
        Amount

    Supports:

        1. pandas DataFrame
        2. Database list of tuples:
           (id, description, amount, category, date)

        Also supports DataFrames using:
            date / amount
            Date / Amount
            DATE / AMOUNT
    """

    if df is None:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Database output
    # --------------------------------------------------------
    #
    # database.get_expenses() returns:
    #
    # [
    #     (id, description, amount, category, date),
    #     ...
    # ]
    #
    # Convert that list into a DataFrame.
    # --------------------------------------------------------

    if isinstance(df, list):

        if not df:
            return pd.DataFrame()

        # Database tuple structure:
        #
        # id, description, amount, category, date

        data = pd.DataFrame(
            df,
            columns=[
                "ID",
                "Description",
                "Amount",
                "Category",
                "Date"
            ]
        )

    elif isinstance(df, pd.DataFrame):

        data = df.copy()

    else:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Normalize column names
    # --------------------------------------------------------

    rename_map = {}

    for column in data.columns:

        column_lower = str(
            column
        ).strip().lower()

        if column_lower == "date":
            rename_map[column] = "Date"

        elif column_lower == "amount":
            rename_map[column] = "Amount"

    data = data.rename(
        columns=rename_map
    )

    return data

# ============================================================
# GET MONTH COUNT
# ============================================================

def get_month_count(df):
    """
    Count monthly spending observations.

    Calendar gaps are included as zero-spending months,
    because the forecasting model works with monthly time
    series data.
    """

    data = prepare_forecast_data(df)

    if data.empty:
        return 0

    if "Date" not in data.columns:
        return 0

    if "Amount" not in data.columns:
        return 0

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

    data = data[data["Amount"] >= 0]

    if data.empty:
        return 0

    monthly = (
        data
        .set_index("Date")
        .resample("MS")["Amount"]
        .sum()
    )

    return len(monthly)


# ============================================================
# AUTOMATIC FORECAST MODEL SELECTION
# ============================================================

def forecast_spending(df):
    """
    Automatically select the appropriate production
    forecasting model.

    Forecasting logic:

        < 4 months
            → Not enough historical data

        4–5 months
            → Statistical forecasting

        >= 6 months
            → Production LSTM IF available

        >= 6 months + production LSTM unavailable
            → Statistical forecasting

        LSTM failure
            → Statistical fallback
    """
    if df is None:
    
        return {
            "success": False,
            "model": "none",
            "months_available": 0,
            "message": "No spending data available."
        }
    
    # ========================================================
    # NORMALIZE DATA
    # ========================================================
    
    data = prepare_forecast_data(df)
    
    if data.empty:
    
        return {
            "success": False,
            "model": "none",
            "months_available": 0,
            "message": "No spending data available."
        }


# ========================================================
 # CHECK NORMALIZED DATA
 # ========================================================

    if data.empty:

        return {
            "success": False,
            "model": "none",
            "months_available": 0,
            "message": "No spending data available."
        }

    # ========================================================
    # VALIDATE REQUIRED COLUMNS
    # ========================================================

    if "Date" not in data.columns:

        return {
            "success": False,
            "model": "none",
            "months_available": 0,
            "message": "Date column not found."
        }

    if "Amount" not in data.columns:

        return {
            "success": False,
            "model": "none",
            "months_available": 0,
            "message": "Amount column not found."
        }
    


    # ========================================================
    # COUNT AVAILABLE MONTHS
    # ========================================================

    month_count = get_month_count(data)

    # ========================================================
    # NOT ENOUGH DATA
    # ========================================================

    if month_count < MIN_STATISTICAL_MONTHS:

        return {
            "success": False,
            "model": "none",
            "months_available": month_count,
            "minimum_required": MIN_STATISTICAL_MONTHS,
            "message": (
                f"At least {MIN_STATISTICAL_MONTHS} months "
                f"of spending history are required for "
                f"forecasting. Currently available: "
                f"{month_count} month(s)."
            )
        }

    # ========================================================
    # LSTM FORECAST
    # ========================================================

    if (
        month_count >= MIN_LSTM_MONTHS
        and PRODUCTION_LSTM_READY
    ):

        try:

            print(
                f"Using production LSTM forecasting "
                f"({month_count} months available)"
            )

            lstm_result = forecast_with_lstm(
                data
            )

            if lstm_result.get("success"):

                return {
                    "success": True,
                    "model": "LSTM",

                    "prediction": float(
                        lstm_result["forecast"]
                    ),

                    "months_used": lstm_result.get(
                        "months_used"
                    ),

                    "next_month": lstm_result.get(
                        "next_month"
                    ),

                    "monthly_data": lstm_result.get(
                        "monthly_data"
                    ),

                    "months_available": month_count,

                    "minimum_required": MIN_LSTM_MONTHS,

                    "fallback": False
                }

            print(
                "Production LSTM did not return "
                "a successful result."
            )

        except Exception as e:

            print(
                "Production LSTM failed. "
                "Falling back to statistical forecasting:"
            )

            print(e)

    # ========================================================
    # STATISTICAL FORECAST
    # ========================================================

    try:

        if month_count >= MIN_LSTM_MONTHS:

            print(
                f"Production LSTM is not currently ready. "
                f"Using statistical forecasting "
                f"({month_count} months available)"
            )

        else:

            print(
                f"Using statistical forecasting "
                f"({month_count} months available)"
            )

        result = forecast_next_month(
            data
        )

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        if result.get("success"):

            return {
                "success": True,

                "model": "Statistical Forecast",

                "prediction": float(
                    result["prediction"]
                ),

                "mae": result.get(
                    "mae"
                ),

                "rmse": result.get(
                    "rmse"
                ),

                "r2": result.get(
                    "r2"
                ),

                "history": result.get(
                    "history"
                ),

                "months_available": month_count,

                "minimum_required": MIN_STATISTICAL_MONTHS,

                "fallback": (
                    month_count >= MIN_LSTM_MONTHS
                    and PRODUCTION_LSTM_READY
                )
            }

        # ----------------------------------------------------
        # STATISTICAL MODEL FAILED
        # ----------------------------------------------------

        return {
            "success": False,
            "model": "none",
            "months_available": month_count,
            "message": (
                "Statistical forecasting "
                "could not generate a prediction."
            )
        }

    # ========================================================
    # UNEXPECTED ERROR
    # ========================================================

    except Exception as e:

        return {
            "success": False,
            "model": "none",
            "months_available": month_count,
            "message": str(e)
        }