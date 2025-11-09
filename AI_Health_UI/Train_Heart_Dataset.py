# ===========================================================
# AI Health Assistant – Heart Disease Detection Trainer
# Author: Akshat Sharma
# Dataset: healthcare_project/heart_dataset.csv
# Description: Trains and saves a production-ready model
# ===========================================================

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# 1. Load Dataset
# -----------------------------------------------------------
df_path = Path("Datasets/heart_dataset.csv")
df = pd.read_csv(df_path)
print("✅ Dataset Loaded Successfully!")
print(f"📊 Shape: {df.shape}")
print("📁 Columns:", list(df.columns))

# -----------------------------------------------------------
# 2. Data Cleaning
# -----------------------------------------------------------
df.replace([np.inf, -np.inf, "?"], np.nan, inplace=True)
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
print("🧹 Cleaned missing values and duplicates.")

# -----------------------------------------------------------
# 3. Define Features and Target
# -----------------------------------------------------------
target_col = "target"  # ensure this matches your dataset
if target_col not in df.columns:
    raise ValueError(f"❌ Target column '{target_col}' not found in dataset!")

X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

# -----------------------------------------------------------
# 4. Handle Missing Values + Feature Scaling
# -----------------------------------------------------------
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------------------------
# 5. Train-Test Split
# -----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print("📚 Data Split Completed.")

# -----------------------------------------------------------
# 6. RandomForest + GridSearchCV
# -----------------------------------------------------------
param_grid = {
    "n_estimators": [100, 150],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "bootstrap": [True]
}

rf = RandomForestClassifier(random_state=42)
grid = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("✅ Best Parameters Found:", grid.best_params_)

# -----------------------------------------------------------
# 7. Evaluate Model
# -----------------------------------------------------------
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Heart Disease Model Accuracy: {acc * 100:.2f}%")
print("\n📈 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧾 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------------------------------------
# 8. Save Model & Preprocessors
# -----------------------------------------------------------
model_dir = Path("AI_Health_UI/models")
model_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(best_model, model_dir / "heart_model.pkl")
joblib.dump(imputer, model_dir / "heart_imputer.pkl")
joblib.dump(scaler, model_dir / "heart_scaler.pkl")

print("\n💾 Heart disease model & preprocessors saved successfully!")

# -----------------------------------------------------------
# 9. Visualization (Optional)
# -----------------------------------------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Reds")
plt.title("💓 Confusion Matrix - Heart Disease Detection")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Feature importance
importances = best_model.feature_importances_
features = df.drop(columns=[target_col]).columns
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features, palette="mako")
plt.title("🔥 Feature Importance - Heart Disease Model")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

print("\n🚀 Training Complete! Ready for Streamlit Integration.")
