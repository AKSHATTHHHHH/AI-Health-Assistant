# ✅ Updated RandomForest Training Script (Python 3.13 Compatible)

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# === Step 1: Load dataset ===
df_path = Path("healthcare_project/merged_clean_health_dataset.csv")
df = pd.read_csv(df_path)
print("📋 Columns in CSV:\n", df.columns)
print("\n🔍 Sample data:\n", df.head())

# === Step 2: Handle missing/infinite values ===
print("\n⚠️ Missing values before cleaning:\n", df.isnull().sum())
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# === Step 3: Encode categorical columns ===
label_cols = df.select_dtypes(include='object').columns
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

# === Step 4: Drop unnecessary columns ===
columns_to_drop = ['Name', 'Doctor', 'Hospital', 'Insurance Provider', 
                   'Date of Admission', 'Discharge Date']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# === Step 5: Split features and target ===
if 'target' not in df.columns:
    raise ValueError("❌ 'target' column not found in dataset!")

X = df.drop('target', axis=1)
y = df['target']

# === Step 6: Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 7: Imputation and scaling ===
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === Step 8: Train model ===
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# === Step 9: Evaluate model ===
y_pred = model.predict(X_test)
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📈 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧾 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === Step 10: Save model, scaler, imputer, feature names ===
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/heart_model.pkl")
joblib.dump(scaler, "models/heart_scaler.pkl")
joblib.dump(imputer, "models/heart_imputer.pkl")
joblib.dump(list(X.columns), "models/feature_names.pkl")
print("💾 Saved: heart_model.pkl, heart_scaler.pkl, heart_imputer.pkl, feature_names.pkl")

# === Step 11: Visualize feature importance ===
feature_importances = model.feature_importances_
feature_names = list(X.columns)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_names, palette="viridis")
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("models/feature_importance.png")
print("📊 Feature importance plot saved as models/feature_importance.png")
