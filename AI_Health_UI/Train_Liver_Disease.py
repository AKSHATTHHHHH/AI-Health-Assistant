# ============================================================
# train_liver_model_fixed.py
# Robust Liver Disease Trainer (Fully Streamlit-Compatible)
# ============================================================
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pandas.api.types import is_numeric_dtype

warnings.filterwarnings("ignore")

# -----------------------------
# CONFIGURATION
# -----------------------------
CSV_PATH = Path("Datasets") / "Liver Patient Dataset (LPD)_train.csv"
OUT_DIR = Path("AI_Health_UI/models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# LOAD CSV (with tolerance)
# -----------------------------
print("📂 Loading CSV:", CSV_PATH)
try:
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
except Exception as e_utf:
    print("utf-8 read failed:", e_utf, "→ retrying with latin1")
    df = pd.read_csv(CSV_PATH, encoding="latin1", on_bad_lines="skip")

print("✅ Initial shape:", df.shape)
print("Raw columns (first 30):", list(df.columns)[:30])

# -----------------------------
# CLEAN COLUMN NAMES
# -----------------------------
def clean_col(c):
    if isinstance(c, str):
        c = c.replace("\u200b", "").replace("\xa0", " ").strip()
        c = pd.Series([c]).str.replace(r"[^\w]+", "_", regex=True).iloc[0]
        return c.strip("_")
    return c

df.columns = [clean_col(c) for c in df.columns]
print("🧹 Cleaned columns (first 30):", list(df.columns)[:30])

# -----------------------------
# IDENTIFY TARGET COLUMN
# -----------------------------
possible_targets = ["Dataset", "Result", "Liver_Disease", "LiverDisease", "Diagnosis", "target"]
target_col = next((t for t in possible_targets if t in df.columns), None)

if target_col is None:
    # try case-insensitive match
    cols_lower = {c.lower(): c for c in df.columns}
    for t in possible_targets:
        if t.lower() in cols_lower:
            target_col = cols_lower[t.lower()]
            break

if target_col is None:
    # fallback: find a binary-looking column (0/1 or two unique values)
    for c in df.columns:
        vals = pd.Series(df[c].dropna().unique()).astype(str).str.strip()
        if vals.nunique() <= 3 and set(vals.unique()).issubset({"0","1","0.0","1.0","true","false","True","False","Yes","No","yes","no"}):
            target_col = c
            print("Inferred target column:", target_col)
            break

if target_col is None:
    raise KeyError("❌ Could not detect target column automatically. Please open CSV and specify the target column name.")

print("🎯 Using target column:", target_col)

# -----------------------------
# FIX TARGET COLUMN FORMAT
# -----------------------------
# common dataset variant: Dataset column coded 1/2 (1 = patient, 2 = healthy etc.)
if target_col in df.columns:
    # normalize to strings then map common truthy values to 1
    df[target_col] = df[target_col].astype(str).str.strip()
    df[target_col] = df[target_col].replace({"2": "0", "No": "0", "no": "0", "False": "0", "false": "0", "N": "0"})
    df[target_col] = df[target_col].replace({"1": "1", "Yes": "1", "yes": "1", "True": "1", "true": "1", "Y": "1"})

# -----------------------------
# SELECT FEATURES (Streamlit order)
# -----------------------------
expected_cols = [
    "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
    "Alkaline_Phosphotase", "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase", "Total_Protiens",
    "Albumin", "Albumin_and_Globulin_Ratio"
]

available = [c for c in expected_cols if c in df.columns]
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    print("⚠️ Missing expected columns in CSV:", missing)
    print("➡️ Proceeding with available columns. If critical features missing, consider cleaning CSV or renaming columns.")

# Keep target column even if it's one of expected_cols
cols_to_use = available.copy()
if target_col not in cols_to_use:
    cols_to_use.append(target_col)

df = df[cols_to_use]
print("✅ Columns being used:", list(df.columns))

# -----------------------------
# ENCODE GENDER (if present)
# -----------------------------
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].astype(str).str.strip().replace({"male": "Male", "female": "Female", "M": "Male", "F": "Female"})
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    # if mapping produced NaN, fill with mode or 0
    if df["Gender"].isna().any():
        mode_gender = df["Gender"].mode()
        fill_val = int(mode_gender.iloc[0]) if not mode_gender.empty else 0
        df["Gender"].fillna(fill_val, inplace=True)

# -----------------------------
# HANDLE MISSING VALUES / TYPE NORMALIZATION
# -----------------------------
df = df.replace("?", np.nan).replace("", np.nan)

# coerce numeric-like columns to numeric where sensible
for col in df.columns:
    # skip target for now (already normalized)
    if col == target_col:
        continue
    # if column looks numeric already, coerce to numeric
    try:
        # If at least half non-null values look numeric, coerce
        sample = df[col].dropna().astype(str).head(200).tolist()
        numeric_like = sum(1 for s in sample if s.replace(".", "", 1).lstrip("-").isdigit())
        if len(sample) > 0 and numeric_like >= max(1, int(len(sample) * 0.5)):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    except Exception:
        pass

# Replace inf / -inf
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Now fill missing values:
for col in df.columns:
    if col == target_col:
        continue
    if is_numeric_dtype(df[col]):
        # numeric: fill with median
        median = df[col].median(skipna=True)
        if pd.isna(median):
            median = 0.0
        df[col].fillna(median, inplace=True)
    else:
        # non-numeric: try to fill with mode, else with placeholder
        mode_vals = df[col].mode(dropna=True)
        if not mode_vals.empty:
            df[col].fillna(mode_vals.iloc[0], inplace=True)
        else:
            df[col].fillna("MISSING", inplace=True)

print("✅ Missing values handled. Any remaining NaNs:", df.isna().sum().sum())

# -----------------------------
# SPLIT FEATURES / TARGET
# -----------------------------
X = df.drop(columns=[target_col])
# map target to 0/1 robustly
def map_target(v):
    try:
        s = str(v).strip().lower()
        if s in {"1", "true", "yes", "y"}:
            return 1
        return 0
    except Exception:
        return 0

y = df[target_col].apply(map_target)

# -----------------------------
# IMPUTE & SCALE NUMERICAL DATA
# -----------------------------
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
print("🔢 Numeric columns:", num_cols)
print("🧾 Categorical/text columns:", cat_cols)

# If categorical columns remain, we leave them as-is (model can accept encoded categories if needed).
# For a robust pipeline, you'd label-encode or one-hot these; here we assume numeric-only model columns.

# Imputer + scaler only if numeric columns exist
imputer = None
scaler = None
if num_cols:
    imputer = SimpleImputer(strategy="mean")
    X[num_cols] = imputer.fit_transform(X[num_cols])

    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
else:
    print("⚠️ No numeric columns detected — skipping imputation/scaling step.")

# For safety, convert any remaining object columns to numeric where possible (coerce), else label-encode simple categories
if cat_cols:
    for c in cat_cols:
        # attempt numeric conversion
        coerced = pd.to_numeric(X[c], errors="coerce")
        if coerced.notna().sum() >= max(1, int(0.5 * len(coerced))):
            X[c] = coerced.fillna(coerced.median())
        else:
            # simple label encoding for small cardinality
            X[c] = X[c].astype(str).fillna("MISSING")
            X[c] = pd.Categorical(X[c]).codes  # -1 for NaN replaced by "MISSING" so no -1s

# -----------------------------
# FINAL SANITY: ensure no NaNs remain
# -----------------------------
if X.isna().sum().sum() > 0:
    print("⚠️ Still found NaNs in X after filling — filling with 0 as last resort.")
    X = X.fillna(0)

# -----------------------------
# SPLIT TRAIN / TEST
# -----------------------------
# If y is all zeros or has too few classes, training will fail - guard for that
if y.nunique() < 2:
    raise ValueError("Target variable has fewer than 2 classes after mapping — cannot train. Inspect target column.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("📊 Train/Test shapes:", X_train.shape, X_test.shape)

# -----------------------------
# TRAIN MODEL (Grid Search)
# -----------------------------
param_grid = {
    "n_estimators": [100, 150],
    "max_depth": [6, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "bootstrap": [True]
}
rf = RandomForestClassifier(random_state=42, class_weight="balanced")
grid = GridSearchCV(rf, param_grid, cv=4, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

best = grid.best_estimator_
print("🏆 Best Parameters:", grid.best_params_)

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = best.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------
# SAVE MODEL & PREPROCESSORS
# -----------------------------
joblib.dump(best, OUT_DIR / "liver_model.pkl")
if imputer is not None:
    joblib.dump(imputer, OUT_DIR / "liver_imputer.pkl")
if scaler is not None:
    joblib.dump(scaler, OUT_DIR / "liver_scaler.pkl")
joblib.dump(list(X.columns), OUT_DIR / "liver_features.pkl")
print("\n💾 Saved model and preprocessors to:", OUT_DIR)
print("🧩 Features saved:", list(X.columns))

# -----------------------------
# VISUALIZATION
# -----------------------------
plt.figure(figsize=(6,4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

try:
    feat_imp = pd.Series(best.feature_importances_, index=X.columns).sort_values(ascending=True)
    plt.figure(figsize=(8,6))
    feat_imp.tail(12).plot(kind="barh")
    plt.title("Feature Importances (Liver Model)")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("⚠️ Could not plot feature importances:", e)

print("\n✅ Training complete. Liver model ready for Streamlit integration.")
