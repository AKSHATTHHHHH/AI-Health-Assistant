# Train_Report_Detector_diabetes_model.py
# Trains a Report Detector (Normal / Abnormal / Inconclusive)
# Drops Doctor/Hospital/Insurance Provider/Billing Amount before training

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ----- CONFIG -----
CSV_PATH = Path("Datasets/Report_Detector_healthcare_dataset.csv")  # change if required
OUT_DIR = Path("AI_Health_UI/models")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUT = OUT_DIR / "report_detector_pipeline.pkl"
LABELMAP_OUT = OUT_DIR / "report_label_map.json"
RANDOM_STATE = 42

# ----- LOAD -----
print("Loading:", CSV_PATH)
df = pd.read_csv(CSV_PATH)
print("Shape:", df.shape)
print("Columns:", list(df.columns))

# ----- DROP PII / MANUAL-FILL FIELDS -----
drop_cols = [
    "Doctor", "Hospital", "Insurance Provider", "Billing Amount",
    "Name", "Date of Admission"
]
existing_drop = [c for c in drop_cols if c in df.columns]
if existing_drop:
    print("Dropping columns (PII/manual-fill):", existing_drop)
    df = df.drop(columns=existing_drop)

# ----- TARGET CHECK -----
target_col = "Test Results"  # adjust only if your file uses another name
if target_col not in df.columns:
    raise KeyError(f"Target column '{target_col}' not found in CSV. Available columns: {list(df.columns)}")

# Keep only rows where target is present
df = df[df[target_col].notna()].copy()

# Standardize target labels and map to integers
label_map = {"Normal": 0, "Abnormal": 1, "Inconclusive": 2}
# If labels use different capitalization/spaces, normalize
df[target_col] = df[target_col].astype(str).str.strip().str.title()
# map; unknown labels will be dropped
df = df[df[target_col].isin(label_map.keys())]
y = df[target_col].map(label_map)

# ----- FEATURES -----
# Remove target from features
X = df.drop(columns=[target_col])

# Drop any columns the user explicitly wants removed or any all-NaN columns
X = X.dropna(axis=1, how="all")

# Identify numeric & categorical columns
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# If no features left, abort
if len(numeric_features) + len(categorical_features) == 0:
    raise ValueError("No features left to train on after preprocessing. Check CSV and drop list.")

# ----- PREPROCESSOR -----
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])


preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
], remainder="drop")  # drop any other columns

# ----- MODEL PIPELINE -----
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clf", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE))
])

# ----- TRAIN/TEST SPLIT -----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# ----- HYPERPARAMETER GRID (small and practical) -----
param_grid = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [None, 10, 20],
    "clf__min_samples_leaf": [1, 2],
    "clf__min_samples_split": [2, 5]
}

grid = GridSearchCV(pipeline, param_grid, cv=4, n_jobs=-1, verbose=1)
print("Starting GridSearchCV...")
grid.fit(X_train, y_train)

best_pipeline = grid.best_estimator_
print("Best params:", grid.best_params_)

# ----- EVALUATION -----
y_pred = best_pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc:.4f}\n")
print("Classification report:\n", classification_report(y_test, y_pred, target_names=list(label_map.keys())))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(label_map.keys()), yticklabels=list(label_map.keys()))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Report Detector")
plt.tight_layout()
plt.show()

# ----- SAVE PIPELINE & LABEL MAP -----
joblib.dump(best_pipeline, MODEL_OUT)
with open(LABELMAP_OUT, "w") as f:
    json.dump(label_map, f)

print(f"Saved pipeline to: {MODEL_OUT}")
print(f"Saved label map to: {LABELMAP_OUT}")

# ----- OPTIONAL: show feature names and importances (best effort) -----
try:
    # Get feature names after preprocessing
    pre = best_pipeline.named_steps["preprocessor"]
    # numeric names
    num_names = numeric_features
    # one-hot names (if any)
    cat_ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    cat_cols = []
    if hasattr(cat_ohe, "get_feature_names_out"):
        cat_cols = list(cat_ohe.get_feature_names_out(categorical_features))
    feature_names = num_names + cat_cols

    # get importances (clf may be wrapped)
    clf = best_pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_") and len(feature_names) == len(clf.feature_importances_):
        importances = clf.feature_importances_
        fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        print("\nTop feature importances:\n", fi.head(20).to_string())
    else:
        print("\nFeature importance not available or length mismatch; skipping.")
except Exception as e:
    print("Could not compute feature importances:", e)

print("\nTraining complete. Use the saved pipeline file in your Streamlit app with joblib.load().")
