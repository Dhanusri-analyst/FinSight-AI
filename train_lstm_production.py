# ============================================================
# FinSight AI — Production LSTM Training
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from modules.database import get_expenses


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/spending_lstm_production.keras"
SCALER_PATH = "models/spending_scaler.pkl"

LOOKBACK = 6

# Minimum history required before attempting production training.
# This is deliberately higher than the inference requirement.
MIN_TRAINING_MONTHS = 12

VALIDATION_MONTHS = 3

EPOCHS = 100
BATCH_SIZE = 4


# ============================================================
# LOAD REAL FINSIGHT DATA
# ============================================================

print("=" * 60)
print("FinSight AI — Production LSTM Training")
print("=" * 60)

rows = get_expenses()

print(f"\nTransactions found: {len(rows)}")

if not rows:
    raise SystemExit(
        "❌ No expense data found in the FinSight database."
    )


# ============================================================
# CONVERT DATABASE DATA TO DATAFRAME
# ============================================================

df = pd.DataFrame(
    rows,
    columns=[
        "ID",
        "Description",
        "Amount",
        "Category",
        "Date"
    ]
)

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df["Amount"] = pd.to_numeric(
    df["Amount"],
    errors="coerce"
)

df = df.dropna(
    subset=["Date", "Amount"]
)

df = df[df["Amount"] >= 0]

if df.empty:
    raise SystemExit(
        "❌ No valid spending data available."
    )


# ============================================================
# AGGREGATE INTO MONTHLY SPENDING
# ============================================================

monthly = (
    df
    .set_index("Date")
    .resample("MS")["Amount"]
    .sum()
    .reset_index()
)

monthly = monthly.sort_values("Date").reset_index(drop=True)


print("\nMonthly spending:")
print(monthly.to_string(index=False))


# ============================================================
# CHECK MINIMUM HISTORY
# ============================================================

if len(monthly) < MIN_TRAINING_MONTHS:
    raise SystemExit(
        f"\n❌ Not enough real historical data for production "
        f"LSTM training.\n"
        f"Minimum required: {MIN_TRAINING_MONTHS} months\n"
        f"Currently available: {len(monthly)} months\n\n"
        f"Add more real spending history before training. "
        f"Do not create synthetic transactions just to train "
        f"the production model."
    )


# ============================================================
# CHRONOLOGICAL TRAIN / VALIDATION SPLIT
# ============================================================

if len(monthly) <= LOOKBACK + VALIDATION_MONTHS:
    raise SystemExit(
        "\n❌ Not enough historical months for a separate "
        "training and validation period."
    )


validation_start = len(monthly) - VALIDATION_MONTHS

train_monthly = monthly.iloc[:validation_start].copy()
validation_monthly = monthly.iloc[validation_start:].copy()


print("\nChronological split:")
print(
    f"Training months:   {len(train_monthly)}"
)
print(
    f"Validation months: {len(validation_monthly)}"
)

print(
    f"Training period:   "
    f"{train_monthly['Date'].min().strftime('%Y-%m')} "
    f"to "
    f"{train_monthly['Date'].max().strftime('%Y-%m')}"
)

print(
    f"Validation period: "
    f"{validation_monthly['Date'].min().strftime('%Y-%m')} "
    f"to "
    f"{validation_monthly['Date'].max().strftime('%Y-%m')}"
)


# ============================================================
# SCALE USING TRAINING DATA ONLY
# ============================================================

train_spending = (
    train_monthly["Amount"]
    .astype(float)
    .values
    .reshape(-1, 1)
)

scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(
    train_spending
)


# ============================================================
# CREATE TRAINING SEQUENCES
# ============================================================

X_train = []
y_train = []

for i in range(LOOKBACK, len(train_scaled)):

    X_train.append(
        train_scaled[i - LOOKBACK:i]
    )

    y_train.append(
        train_scaled[i]
    )

X_train = np.array(X_train)
y_train = np.array(y_train)


if len(X_train) == 0:
    raise SystemExit(
        "\n❌ No training sequences could be created."
    )


print("\nTraining data shape:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)


# ============================================================
# CREATE VALIDATION SEQUENCES
# ============================================================

# Validation sequences need the final LOOKBACK training
# observations as context for the first validation month.

combined_for_validation = pd.concat(
    [
        train_monthly.tail(LOOKBACK),
        validation_monthly
    ],
    ignore_index=True
)

combined_values = (
    combined_for_validation["Amount"]
    .astype(float)
    .values
    .reshape(-1, 1)
)

combined_scaled = scaler.transform(
    combined_values
)

X_validation = []
y_validation = []

for i in range(
    LOOKBACK,
    len(combined_scaled)
):

    X_validation.append(
        combined_scaled[i - LOOKBACK:i]
    )

    y_validation.append(
        combined_scaled[i]
    )

X_validation = np.array(X_validation)
y_validation = np.array(y_validation)


if len(X_validation) == 0:
    raise SystemExit(
        "\n❌ No validation sequences could be created."
    )


print("\nValidation data shape:")
print("X_validation:", X_validation.shape)
print("y_validation:", y_validation.shape)


# ============================================================
# BUILD LSTM MODEL
# ============================================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(LOOKBACK, 1)
    ),

    tf.keras.layers.LSTM(
        64,
        return_sequences=True
    ),

    tf.keras.layers.Dropout(
        0.2
    ),

    tf.keras.layers.LSTM(
        32
    ),

    tf.keras.layers.Dropout(
        0.2
    ),

    tf.keras.layers.Dense(
        16,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        1
    )
])


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nStarting LSTM training...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=False,
    verbose=1
)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

validation_predictions_scaled = model.predict(
    X_validation,
    verbose=0
)


# Convert predictions and actual values back to rupees.

validation_predictions = scaler.inverse_transform(
    validation_predictions_scaled
).flatten()

validation_actual = scaler.inverse_transform(
    y_validation.reshape(-1, 1)
).flatten()


# ============================================================
# VALIDATION METRICS
# ============================================================

mae = mean_absolute_error(
    validation_actual,
    validation_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        validation_actual,
        validation_predictions
    )
)


print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print(
    f"Validation MAE:  ₹{mae:,.2f}"
)

print(
    f"Validation RMSE: ₹{rmse:,.2f}"
)


# ============================================================
# CREATE MODELS DIRECTORY
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)


# ============================================================
# SAVE PRODUCTION MODEL
# ============================================================

model.save(
    MODEL_PATH
)

print(
    f"\n✅ Production LSTM saved to: "
    f"{MODEL_PATH}"
)


# ============================================================
# SAVE SCALER
# ============================================================

joblib.dump(
    scaler,
    SCALER_PATH
)

print(
    f"✅ Production scaler saved to: "
    f"{SCALER_PATH}"
)


# ============================================================
# TRAINING SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Total months:       {len(monthly)}"
)

print(
    f"Training months:    {len(train_monthly)}"
)

print(
    f"Validation months:  {len(validation_monthly)}"
)

print(
    f"Training sequences: {len(X_train)}"
)

print(
    f"Lookback:           {LOOKBACK} months"
)

print(
    f"Validation MAE:     ₹{mae:,.2f}"
)

print(
    f"Validation RMSE:    ₹{rmse:,.2f}"
)

print(
    "\n⚠️ Production LSTM is saved, "
    "but PRODUCTION_LSTM_READY remains False "
    "until final inference validation is completed."
)