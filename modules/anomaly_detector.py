import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_spending_anomalies(df, contamination=None):
    """Detect unusual transaction amounts using Isolation Forest.

    Returns a copy of the dataframe with:
      - Is_Anomaly: True for unusual transactions
      - Anomaly_Score: higher values mean more normal; lower values are more unusual

    The model uses transaction amount (log-transformed) as the feature so very
    large amounts do not dominate the model because of scale.
    """
    result = df.copy()

    if result.empty or "Amount" not in result.columns:
        result["Is_Anomaly"] = False
        result["Anomaly_Score"] = 0.0
        return result

    result["Amount"] = pd.to_numeric(result["Amount"], errors="coerce")
    valid_mask = result["Amount"].notna() & (result["Amount"] > 0)

    result["Is_Anomaly"] = False
    result["Anomaly_Score"] = 0.0

    valid = result.loc[valid_mask].copy()

    # Isolation Forest needs more than a couple of observations to be useful.
    if len(valid) < 5:
        return result

    amounts = valid[["Amount"]].copy()
    amounts["Amount"] = amounts["Amount"].apply(lambda x: __import__("math").log1p(x))

    if contamination is None:
        # With small personal-finance datasets, aim to flag roughly one
        # unusual transaction while avoiding an overly aggressive detector.
        contamination = min(0.20, max(1.0 / len(valid), 0.05))

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42
    )

    predictions = model.fit_predict(amounts)
    scores = model.decision_function(amounts)

    result.loc[valid.index, "Is_Anomaly"] = predictions == -1
    result.loc[valid.index, "Anomaly_Score"] = scores

    return result