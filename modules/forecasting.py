import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def forecast_next_month(df):
    """
    Forecast next month's total spending from monthly history.

    Uses:
        - lag-1
        - lag-2
        - lag-3
        - 3-month rolling mean

    At least 4 calendar months are required.

    For 4 months of history, there is only one supervised
    training row, so evaluation metrics are not calculated.

    For 5+ months, the last 20% of supervised observations
    are held out for evaluation.
    """

    # ========================================================
    # VALIDATE INPUT
    # ========================================================

    if (
        df is None
        or df.empty
        or "Date" not in df.columns
        or "Amount" not in df.columns
    ):
        return {
            "success": False,
            "message": "Date and Amount columns are required."
        }

    # ========================================================
    # CLEAN DATA
    # ========================================================

    work = df[
        ["Date", "Amount"]
    ].copy()

    work["Date"] = pd.to_datetime(
        work["Date"],
        errors="coerce"
    )

    work["Amount"] = pd.to_numeric(
        work["Amount"],
        errors="coerce"
    )

    work = work.dropna(
        subset=["Date", "Amount"]
    )

    work = work[
        work["Amount"] >= 0
    ]

    if work.empty:
        return {
            "success": False,
            "message": "No valid spending data available."
        }

    # ========================================================
    # MONTHLY AGGREGATION
    # ========================================================

    monthly = (
        work
        .set_index("Date")["Amount"]
        .resample("MS")
        .sum()
        .reset_index()
    )

    if len(monthly) < 4:
        return {
            "success": False,
            "message": (
                "At least 4 months of spending "
                "history are required."
            )
        }

    # ========================================================
    # FILL CALENDAR GAPS
    # ========================================================

    full_index = pd.date_range(
        monthly["Date"].min(),
        monthly["Date"].max(),
        freq="MS"
    )

    monthly = (
        monthly
        .set_index("Date")
        .reindex(
            full_index,
            fill_value=0
        )
        .rename_axis("Date")
        .reset_index()
    )

    # ========================================================
    # CREATE FEATURES
    # ========================================================

    monthly["lag1"] = (
        monthly["Amount"].shift(1)
    )

    monthly["lag2"] = (
        monthly["Amount"].shift(2)
    )

    monthly["lag3"] = (
        monthly["Amount"].shift(3)
    )

    monthly["rolling3"] = (
        monthly["Amount"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    supervised = monthly.dropna().copy()

    if supervised.empty:
        return {
            "success": False,
            "message": "Not enough data to create forecast features."
        }

    features = [
        "lag1",
        "lag2",
        "lag3",
        "rolling3"
    ]

    X = supervised[features]
    y = supervised["Amount"]

    # ========================================================
    # EVALUATION
    # ========================================================

    mae = float("nan")
    rmse = float("nan")
    r2 = float("nan")

    # --------------------------------------------------------
    # 4 MONTHS
    # --------------------------------------------------------
    #
    # Exactly 4 months produces only one supervised row.
    # There is not enough data for a meaningful train/test
    # evaluation split.
    #
    # We therefore skip evaluation and train the model on
    # the available supervised observation.
    # --------------------------------------------------------

    if len(supervised) == 1:

        evaluation_possible = False

    else:

        evaluation_possible = True

        test_size = max(
            1,
            int(
                np.ceil(
                    len(supervised) * 0.2
                )
            )
        )

        # Make sure at least one training observation exists.
        if len(supervised) - test_size >= 1:

            X_train = X.iloc[
                :-test_size
            ]

            X_test = X.iloc[
                -test_size:
            ]

            y_train = y.iloc[
                :-test_size
            ]

            y_test = y.iloc[
                -test_size:
            ]

            evaluation_model = LinearRegression()

            evaluation_model.fit(
                X_train,
                y_train
            )

            predictions = (
                evaluation_model.predict(
                    X_test
                )
            )

            mae = float(
                mean_absolute_error(
                    y_test,
                    predictions
                )
            )

            rmse = float(
                np.sqrt(
                    mean_squared_error(
                        y_test,
                        predictions
                    )
                )
            )

            if len(y_test) >= 2:

                r2 = float(
                    r2_score(
                        y_test,
                        predictions
                    )
                )

    # ========================================================
    # FINAL MODEL
    # ========================================================

    final_model = LinearRegression()

    final_model.fit(
        X,
        y
    )

    # ========================================================
    # PREPARE NEXT-MONTH FEATURES
    # ========================================================

    last = monthly.iloc[-1]

    next_features = pd.DataFrame([
        {
            "lag1": last["Amount"],

            "lag2": monthly.iloc[-2][
                "Amount"
            ],

            "lag3": monthly.iloc[-3][
                "Amount"
            ],

            "rolling3": monthly.iloc[-3:][
                "Amount"
            ].mean()
        }
    ])[features]

    # ========================================================
    # FORECAST
    # ========================================================

    prediction = float(
        final_model.predict(
            next_features
        )[0]
    )

    prediction = max(
        0.0,
        prediction
    )

    # ========================================================
    # RETURN
    # ========================================================

    return {
        "success": True,

        "prediction": prediction,

        "mae": mae,

        "rmse": rmse,

        "r2": r2,

        "history": monthly[
            ["Date", "Amount"]
        ].copy(),

        "evaluation_available": evaluation_possible
    }