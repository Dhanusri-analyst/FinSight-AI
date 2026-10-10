import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# FinSight AI - ML Model Training
# ==========================================

DATA_FILE = "data/cleaned_budgetwise.csv"
MODEL_DIR = "models"

print("Loading cleaned dataset...")

df = pd.read_csv(DATA_FILE)

# ------------------------------------------
# Prepare text data
# ------------------------------------------

df["notes"] = df["notes"].fillna("").astype(str)

# Use notes as the main text feature
X = df["notes"]

# Category is what the model must learn
y = df["category"]

print("Total transactions:", len(df))
print("Number of categories:", y.nunique())

# ------------------------------------------
# Train/Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ------------------------------------------
# TF-IDF
# ------------------------------------------

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    max_features=10000,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF features:", X_train_tfidf.shape[1])

# ------------------------------------------
# Train Logistic Regression
# ------------------------------------------

print("\nTraining ML model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)

# ------------------------------------------
# Evaluate
# ------------------------------------------

print("\nEvaluating model...")

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\n==========================================")
print("MODEL RESULTS")
print("==========================================")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ------------------------------------------
# Save model
# ------------------------------------------

os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, "expense_model.pkl")
vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print("\n==========================================")
print("MODEL SAVED")
print("==========================================")
print("Model:", model_path)
print("Vectorizer:", vectorizer_path)

# ------------------------------------------
# Test sample predictions
# ------------------------------------------

test_expenses = [
    "Uber ride to college",
    "Amazon laptop purchase",
    "Electricity payment",
    "School tuition fee",
    "Monthly house rent",
    "Movie tickets with friends",
    "Doctor consultation",
    "Salary credited",
    "Freelance project payment"
]

print("\nSample predictions:")

test_vectors = vectorizer.transform(test_expenses)
predictions = model.predict(test_vectors)

for expense, prediction in zip(test_expenses, predictions):
    print(f"{expense} -> {prediction}")
