# ===========================================================
# AI Health Assistant – Optimized Diabetes Model Trainer
# Author: Akshat Sharma
# ===========================================================

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# ===========================================================
# ⚙️ CONFIG
# ===========================================================
DATASET_PATH = Path("Datasets/Diabetes.csv")
MODEL_DIR = Path("AI_Health_UI/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ===========================================================
# 🧠 TRAIN DIABETES MODEL
# ===========================================================
def train_large_diabetes_model():
    print("📂 Loading dataset...")
    # Efficient read for large CSVs
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    print(f"✅ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

    # Fix weird column names (strip spaces etc.)
    df.columns = df.columns.str.strip()

    # Detect target column automatically
    target_cols = [c for c in df.columns if "diabetes" in c.lower()]
    if not target_cols:
        raise ValueError("❌ Target column not found (expected Diabetes_binary or Outcome).")
    target_col = target_cols[0]
    print(f"🎯 Target column detected: {target_col}")

    # Drop unnecessary columns (like ID or duplicates)
    df = df.loc[:, ~df.columns.duplicated()]

    # Clean missing & infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=0, how="any", inplace=True)
    print(f"🧹 Cleaned dataset shape: {df.shape}")

    # Split X and y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Ensure numeric conversion
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean())

    # Impute + scale
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train with optimized RandomForest
    print("🚀 Training model (this may take a few minutes)...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,  # use all CPU cores
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Accuracy: {acc * 100:.2f}%")
    print("\n📈 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save trained artifacts
    joblib.dump(model, MODEL_DIR / "diabetes_model.pkl")
    joblib.dump(imputer, MODEL_DIR / "diabetes_imputer.pkl")
    joblib.dump(scaler, MODEL_DIR / "diabetes_scaler.pkl")

    print("\n💾 Model, imputer, and scaler saved successfully in:")
    print(MODEL_DIR.resolve())
    print("\n✅ Diabetes model training completed!\n")


# ===========================================================
# MAIN
# ===========================================================
if __name__ == "__main__":
    train_large_diabetes_model()