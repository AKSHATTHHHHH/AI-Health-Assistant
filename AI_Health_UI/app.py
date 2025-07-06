# Final AI Health Assistant App with Enhanced UI, OCR, and Readable Outputs

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
import altair as alt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

def generate_pdf_report(prediction, rule_diag, proba, input_data):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)

    y = 750
    c.drawString(50, y, "🧠 AI Health Diagnosis Report")
    y -= 30
    c.drawString(50, y, f"Prediction: {'Heart Disease' if prediction else 'Healthy'}")
    y -= 20
    c.drawString(50, y, f"Confidence: {proba:.2%}")
    y -= 20
    c.drawString(50, y, f"Rule-Based Insight: {rule_diag}")
    y -= 30
    c.drawString(50, y, "Patient Inputs:")

    for key, val in input_data.items():
        y -= 20
        c.drawString(70, y, f"{key}: {val}")
        if y < 100:
            c.showPage()
            y = 750

    c.save()
    buffer.seek(0)
    return buffer

# Configure Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Page Config
st.set_page_config(page_title="🧠 AI Health Assistant", layout="wide", page_icon="🩺")

# Sidebar Branding
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3771/3771395.png", width=60)
    st.markdown("### 🌐 AI-Powered Structure Health Monitoring")
    st.markdown("""
Welcome to your **Smart Diagnosis** platform.

**Powered by**  
- 🧠 Streamlit  
- 🤖 Machine Learning  
- 📝 OCR for Reports  
- 📊 Visual Analytics
    """)
    st.divider()
    st.markdown("### 🔧 Settings")
    show_confidence = st.checkbox("📊 Show Prediction Confidence", value=True)
    show_ocr = st.checkbox("📑 Enable OCR Preview", value=True)
    enable_debug = st.checkbox("🛠️ Developer Debug Info", value=False)

# Load model and dataset
@st.cache_data
def load_model_and_data():
    model = joblib.load("model.pkl")
    df = pd.read_csv("healthcare_project/merged_clean_health_dataset.csv")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=0, how='all', inplace=True)

    drop_cols = ['Name', 'Doctor', 'Hospital', 'Insurance Provider', 'Date of Admission', 'Discharge Date']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category').cat.codes

    X = df.drop('target', axis=1)
    y = df['target']

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return model, scaler, X, y, df

model, scaler, X_full, y_full, df_full = load_model_and_data()

# Mapping for readable fields
def readable_value(key, val):
    if key.lower() == 'gender':
        return 'Male' if val == 1 else 'Female'
    elif key.lower() == 'blood type':
        return ['A', 'B', 'AB', 'O'][int(val) % 4] if isinstance(val, (int, float)) else val
    elif key.lower() == 'age':
        if val < 18:
            return "Child"
        elif val < 40:
            return "Young Adult"
        elif val < 60:
            return "Adult"
        else:
            return "Senior"
    return val

# Tabs Layout
st.title("🧠 AI-Powered Medical Report Detection Tool")
tab1, tab2, tab3 = st.tabs(["🔍 Diagnose", "📊 Charts", "ℹ️ About"])

# --- Diagnose Tab ---
with tab1:
    st.header("📋 Patient Diagnosis Panel")
    st.caption("Fill out patient data and upload a medical report for AI-powered diagnosis.")

    input_data = {}
    cols = df_full.drop(columns='target').columns.tolist()
    with st.form("patient_form"):
        for i, col in enumerate(cols):
            if col.lower() == "sex":
                sex_display = st.selectbox("Sex", ["Male", "Female"], key=f"input_{i}")
                input_data[col] = 1 if sex_display == "Male" else 0
            else:
                input_data[col] = st.number_input(f"{col}", value=0.0, key=f"input_{i}")
        submitted = st.form_submit_button("🔍 Diagnose")

    uploaded_file = st.file_uploader("📤 Upload Medical Report (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])
    extracted_text = ""

    if uploaded_file:
        st.success(f"Uploaded: {uploaded_file.name}")

        def extract_text(file):
            if file.type == "application/pdf":
                images = convert_from_bytes(file.read())
                return "\n".join([pytesseract.image_to_string(img) for img in images])
            elif file.type.startswith("image"):
                return pytesseract.image_to_string(Image.open(file))
            return "Unsupported file."

        extracted_text = extract_text(uploaded_file)
        if show_ocr:
            st.text_area("🧾 Extracted Text from Report", value=extracted_text, height=200)

    if submitted:
        with st.spinner("Analyzing patient data..."):
            user_df = pd.DataFrame([input_data])
            user_scaled = scaler.transform(user_df)
            prediction = model.predict(user_scaled)[0]
            proba = model.predict_proba(user_scaled)[0][1]

            chol = input_data.get('chol', 0)
            thalach = input_data.get('thalach', 0)
            fbs = input_data.get('fbs', 0)
            rule_diag = "High Cholesterol" if chol > 240 else "Low Heart Rate - Risk" if thalach < 100 else "Possible Diabetes" if fbs == 1 else "Normal"

        st.subheader("🩺 AI Diagnosis Result")
        if prediction == 0:
            st.success("✅ Heart is likely healthy.")
        else:
            st.error("⚠️ Risk of Heart Disease.")

        if show_confidence:
            st.metric("📊 Prediction Confidence", f"{proba:.2%}")
        st.info(f"💡 Rule-Based Insight: {rule_diag}")

        st.markdown("### 📑 PDF Report")
        pdf_bytes = generate_pdf_report(prediction, rule_diag, proba, input_data)
        st.download_button("📥 Download PDF Report", data=pdf_bytes, file_name="diagnosis_report.pdf", mime="application/pdf")

# --- Charts Tab ---
with tab2:
    st.header("📊 Visual Summary & Insights")
    col1, col2 = st.columns(2)

    gender_filter = col1.selectbox("Filter by Gender", ["All"] + list(map(str, df_full['Gender'].unique())) if 'Gender' in df_full.columns else ["All"])
    cond_filter = col2.selectbox("Filter by Condition", ["All"] + list(map(str, df_full['Medical Condition'].unique())) if 'Medical Condition' in df_full.columns else ["All"])

    df_vis = df_full.copy()
    if gender_filter != "All": df_vis = df_vis[df_vis['Gender'] == float(gender_filter)]
    if cond_filter != "All": df_vis = df_vis[df_vis['Medical Condition'] == float(cond_filter)]

    df_vis['Diagnosis'] = df_vis.apply(lambda row: "High Cholesterol" if row['chol'] > 240 else "Low Heart Rate - Risk" if row['thalach'] < 100 else "Possible Diabetes" if row['fbs'] == 1 else "Normal", axis=1)

    st.subheader("📌 Diagnosis Categories (Rule-Based)")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='Diagnosis', data=df_vis, palette='Set2', ax=ax1)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    df_vis['Diagnosis'].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('Set2'), ax=ax2)
    ax2.set_ylabel('')
    st.pyplot(fig2)

    st.markdown("ℹ️ These insights are generated from rule-based thresholds, not ML output.")

    if hasattr(model, "feature_importances_"):
        st.subheader("🧬 Feature Importance")
        feat_df = pd.DataFrame({
            "Feature": df_full.drop('target', axis=1).columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        chart = alt.Chart(feat_df).mark_bar().encode(
            x=alt.X("Importance:Q", title="Importance"),
            y=alt.Y("Feature:N", sort='-x'),
            tooltip=['Feature', 'Importance']
        ).properties(width=600)
        st.altair_chart(chart)
    else:
        st.warning("⚠️ Feature importance not available for this model.")

# --- About Tab ---
with tab3:
    st.header("ℹ️ About This App")
    st.markdown("""
This tool demonstrates AI integration in health diagnostics using real-time patient data.

**Key Features:**
- 🧠 ML prediction for heart conditions using supervised learning
- 📑 OCR-based medical report processing
- 📊 Rule-based insight detection
- 📈 Visual analytics & filtering
- 📤 Report uploads with optional text extraction
- 📥 Exportable PDF/text output

**Tech Stack:**
Python · Streamlit · scikit-learn · Tesseract OCR · Altair · Matplotlib · Seaborn

**Version:** v1.0.0  
**Developed By:** Akshat Sharma  
**Mentor:** Prof. XYZ  
**Email:** [akshatsharma3@shooliniuniversity.com](mailto:akshatsharma3@shooliniuniversity.com)

**Disclaimer:** For educational & research use only.
    """)