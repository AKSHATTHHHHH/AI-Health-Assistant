import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === Step 1: Load dataset ===
df = pd.read_csv("healthcare_project/merged_clean_health_dataset.csv")
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
for col in columns_to_drop:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

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
model = LogisticRegression()
model.fit(X_train, y_train)

# === Step 9: Evaluate model ===
y_pred = model.predict(X_test)
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📈 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧾 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === Step 10: Rule-based diagnosis ===
def ai_diagnosis(row):
    if row['chol'] > 240:
        return "High Cholesterol"
    elif row['thalach'] < 100:
        return "Low Heart Rate - Possible Risk"
    elif row['fbs'] == 1:
        return "Possible Diabetes"
    else:
        return "Normal"

df['Diagnosis'] = df.apply(ai_diagnosis, axis=1)
print("\n🔬 Sample Rule-Based Diagnoses:\n", df[['age', 'chol', 'thalach', 'fbs', 'Diagnosis']].head())

# === Step 11: Charts ===
sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.countplot(x='Diagnosis', data=df, palette='Set2')
plt.title('Diagnosis Distribution')
plt.xlabel('Diagnosis Category')
plt.ylabel('Number of Patients')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
df['Diagnosis'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Set2'))
plt.title('Diagnosis Proportion')
plt.ylabel('')
plt.tight_layout()
plt.show()

# === Step 12: Save model and features ===
joblib.dump(model, "model.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")
print("💾 Model and feature list saved: model.pkl & feature_names.pkl")
