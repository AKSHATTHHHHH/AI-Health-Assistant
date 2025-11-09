# ===========================================
# 🩺 AI Health Assistant v3.5 - FINAL IoT + Firestore
# Developer: Akshat Sharma
# Features: IoT live input + Manual input, DL + SKLearn, OCR + NLP, Reports
# ===========================================

import os
from pathlib import Path
from datetime import datetime
import json
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import random
import time
import firebase_admin
from firebase_admin import credentials, firestore

# -------------------------
# FIREBASE SETUP
# -------------------------
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore

# Load Firebase credentials from Streamlit secrets
firebase_config = st.secrets["FIREBASE"]

cred = credentials.Certificate(firebase_config)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Alternatively, if using a local JSON file for credentials:
#
cred = credentials.Certificate("AI_Health_UI/ai-power-structural-health-m-s-firebase-adminsdk-fbsvc-065dfad472.json")  # <-- Set path here
firebase_admin.initialize_app(cred)
db = firestore.client()

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="AI Health Assistant", page_icon="🩺", layout="wide")
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DISEASES = ["Heart", "Diabetes", "Liver", "Blood"]

# -------------------------
# DEPENDENCIES CHECK
# -------------------------
KERAS_AVAILABLE = False
try:
    import importlib
    tf_keras = importlib.import_module("tensorflow.keras.models")
    keras_load_model = tf_keras.load_model
    KERAS_AVAILABLE = True
except Exception:
    try:
        keras = importlib.import_module("keras.models")
        keras_load_model = keras.load_model
        KERAS_AVAILABLE = True
    except Exception:
        KERAS_AVAILABLE = False

OCR_AVAILABLE = False
try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

NLP_AVAILABLE = False
try:
    from transformers import pipeline
    NLP_AVAILABLE = True
except Exception:
    NLP_AVAILABLE = False

# -------------------------
# MODEL LOADING HELPERS
# -------------------------
def load_sklearn_joblib(path: Path):
    try: return joblib.load(path)
    except Exception: return None

def load_keras_model(path: Path):
    if not KERAS_AVAILABLE: return None
    try: return keras_load_model(str(path))
    except: return None

def load_model_bundle(disease: str):
    bundle = {}
    model_pkl = MODELS_DIR / f"{disease.lower()}_model.pkl"
    imputer_pkl = MODELS_DIR / f"{disease.lower()}_imputer.pkl"
    scaler_pkl = MODELS_DIR / f"{disease.lower()}_scaler.pkl"
    model_h5 = MODELS_DIR / f"{disease.lower()}_model_keras.h5"

    if model_pkl.exists():
        bundle["model"] = load_sklearn_joblib(model_pkl)
        bundle["type"] = "sklearn"
    elif model_h5.exists():
        model = load_keras_model(model_h5)
        if model:
            bundle["model"] = model
            bundle["type"] = "keras"
        else: return None
    else:
        return None

    bundle["imputer"] = load_sklearn_joblib(imputer_pkl) if imputer_pkl.exists() else None
    bundle["scaler"] = load_sklearn_joblib(scaler_pkl) if scaler_pkl.exists() else None
    return bundle

model_bundles = {d: load_model_bundle(d) for d in DISEASES}

# -------------------------
# PREPROCESS & PREDICT
# -------------------------
def preprocess_input(bundle, df_input: pd.DataFrame):
    arr = df_input.values.astype(float)
    imputer, scaler = bundle.get("imputer"), bundle.get("scaler")
    if imputer: arr = imputer.transform(df_input)
    if scaler: arr = scaler.transform(arr)
    return arr

def predict_model(bundle, arr, show_confidence=True):
    model_type = bundle["type"]
    model = bundle["model"]
    conf = None

    if model_type == "sklearn":
        pred = model.predict(arr)[0]
        if show_confidence and hasattr(model, "predict_proba"): conf = model.predict_proba(arr)[0]
    else:
        probs = model.predict(arr)
        if probs.ndim == 2 and probs.shape[1] > 1:
            pred = int(np.argmax(probs[0]))
            conf = probs[0]
        elif probs.ndim == 2 and probs.shape[1] == 1:
            prob = float(probs[0][0])
            pred = 1 if prob >= 0.5 else 0
            conf = np.array([1-prob, prob])
        else:
            pred = int(np.round(float(probs[0][0])))
    return pred, conf

# -------------------------
# FIRESTORE INTEGRATION
# -------------------------
def save_prediction_history(disease, columns, input_values, pred, conf):
    record = {
        "Date": datetime.now().isoformat(),
        "Disease": disease,
        "Input": {c: v for c, v in zip(columns, input_values)},
        "Prediction": int(pred),
        "Confidence": float(np.max(conf)) if conf is not None else None
    }
    db.collection("predictions").add(record)

def generate_report(disease, columns, input_values, pred, conf):
    conf_str = f"{np.max(conf):.2f}" if conf is not None else "N/A"
    status = "⚠️ Likely condition detected." if pred else "💚 No condition detected."
    report = f"---\nDisease: {disease}\nStatus: {status}\nConfidence: {conf_str}\nInput Data:\n"
    for col, val in zip(columns, input_values): report += f"  {col}: {val}\n"
    report += "---"
    return report

# -------------------------
# OCR & NLP
# -------------------------
@st.cache_resource
def load_nlp_model():
    if NLP_AVAILABLE:
        try: return pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
        except: return None
    return None

nlp_model = load_nlp_model()

def extract_text_from_pdf_util(file_path: Path) -> str:
    if not OCR_AVAILABLE: return "OCR libraries not available."
    text = ""
    images = convert_from_path(str(file_path))
    for img in images: text += pytesseract.image_to_string(img)
    return text

# -------------------------
# STREAMLIT UI
# -------------------------
st.sidebar.title("AI Health Assistant 🩺 v3.5")
disease_selection = st.sidebar.selectbox("Select Disease Module", DISEASES)
show_confidence = st.sidebar.checkbox("Show Confidence", value=True)
enable_nlp = st.sidebar.checkbox("Enable Report NLP", value=True)
enable_ocr = st.sidebar.checkbox("Enable OCR", value=True)
enable_iot = st.sidebar.checkbox("Enable Live IoT Input", value=False)

st.title(f"🩺 {disease_selection} Diagnosis & Analysis")
st.markdown("---")

# ABOUT SECTION
about_sections = {
    "Heart":"Heart Disease Module predicts likelihood of cardiovascular problems based on patient parameters.",
    "Diabetes":"Diabetes Module predicts risk of diabetes based on glucose levels, BMI, insulin, and other metrics.",
    "Liver":"Liver Module predicts potential liver disease using bilirubin, liver enzymes, proteins, etc.",
    "Blood":"Blood/Report Module analyzes RBC, WBC, Hemoglobin for anomalies or disease risk."
}
st.info(about_sections.get(disease_selection,""))

# -------------------------
# IoT Live Input Simulation
# -------------------------
def fetch_iot_data(disease):
    if disease=="Heart": return {k: random.randint(0,10) if k!="oldpeak" else round(random.uniform(0,5),1) for k in ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]}
    elif disease=="Diabetes": return {k: random.randint(0,200) if k!="BMI" else round(random.uniform(18,40),1) for k in ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]}
    elif disease=="Liver": return {k: random.randint(0,150) if k not in ["Age","Total_Bilirubin","Direct_Bilirubin","Total_Protiens","Albumin","Albumin_and_Globulin_Ratio"] else round(random.uniform(0,10),1) for k in ["Age","Gender","Total_Bilirubin","Direct_Bilirubin","Alkaline_Phosphotase","Alamine_Aminotransferase","Aspartate_Aminotransferase","Total_Protiens","Albumin","Albumin_and_Globulin_Ratio"]}
    elif disease=="Blood": return {"Hemoglobin": round(random.uniform(10,18),1),"RBC": round(random.uniform(3.5,6),2),"WBC": round(random.uniform(4,12),1)}
    else: return {}

# -------------------------
# INPUT UI
# -------------------------
def disease_inputs_ui(disease):
    inputs, columns = [], []
    if disease=="Heart":
        age = st.number_input("Age",1,120,45); sex = st.selectbox("Sex",["Male","Female"])
        cp = st.selectbox("Chest Pain Type", [0,1,2,3]); trestbps = st.number_input("BP",80,200,130)
        chol = st.number_input("Cholesterol",100,400,200); fbs = st.selectbox("Fasting Sugar>120", [0,1])
        restecg = st.selectbox("ECG Results", [0,1,2]); thalach = st.number_input("Max HR",50,250,150)
        exang = st.selectbox("Exercise Induced Angina",[0,1]); oldpeak = st.number_input("ST Depression",0.0,6.5,1.0,0.1)
        slope = st.selectbox("Slope ST",[0,1,2]); ca = st.number_input("Major Vessels",0,3,0); thal = st.selectbox("Thalassemia",[1,2,3])
        columns = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
        inputs = [age,1 if sex=="Male" else 0,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
    elif disease=="Diabetes":
        pregnancies=st.number_input("Pregnancies",0,20,0); glucose=st.number_input("Glucose",0,300,120)
        bp=st.number_input("BP",0,200,70); skin=st.number_input("Skin Thickness",0,100,20)
        insulin=st.number_input("Insulin",0,900,80); bmi=st.number_input("BMI",0.0,70.0,28.0)
        dpf=st.number_input("Diabetes Pedigree",0.0,2.5,0.5); age=st.number_input("Age",1,120,35)
        columns=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
        inputs=[pregnancies,glucose,bp,skin,insulin,bmi,dpf,age]
    elif disease=="Liver":
        age_v=st.number_input("Age",1,120,45); gender=st.selectbox("Gender",["Male","Female"])
        total_bilirubin=st.number_input("Total Bilirubin",0.0,10.0,1.0); direct_bilirubin=st.number_input("Direct Bilirubin",0.0,5.0,0.2)
        alk_phos=st.number_input("ALP",0,1500,100); sgpt=st.number_input("SGPT",0,1000,30); sgot=st.number_input("SGOT",0,1000,30)
        total_proteins=st.number_input("Total Proteins",0.0,10.0,7.0); albumin=st.number_input("Albumin",0.0,6.0,3.5); ag_ratio=st.number_input("Albumin/Globulin Ratio",0.0,3.0,1.0)
        columns=["Age","Gender","Total_Bilirubin","Direct_Bilirubin","Alkaline_Phosphotase","Alamine_Aminotransferase","Aspartate_Aminotransferase","Total_Protiens","Albumin","Albumin_and_Globulin_Ratio"]
        inputs=[age_v,1 if gender=="Male" else 0,total_bilirubin,direct_bilirubin,alk_phos,sgpt,sgot,total_proteins,albumin,ag_ratio]
    elif disease=="Blood":
        hb=st.number_input("Hemoglobin",0.0,25.0,13.0); rbc=st.number_input("RBC",0.0,10.0,4.5); wbc=st.number_input("WBC",0.0,50.0,6.0)
        columns=["Hemoglobin","RBC","WBC"]; inputs=[hb,rbc,wbc]
    return inputs, columns

# -------------------------
# Use IoT if enabled
# -------------------------
if enable_iot:
    iot_data = fetch_iot_data(disease_selection)
    st.info(f"🔴 Live IoT Data: {iot_data}")
    columns = list(iot_data.keys())
    inputs = list(iot_data.values())
else:
    inputs, columns = disease_inputs_ui(disease_selection)

# -------------------------
# PREDICTION
# -------------------------
if st.button(f"🔍 Predict {disease_selection}"):
    bundle = model_bundles.get(disease_selection)
    if not bundle: st.error(f"{disease_selection} model missing.")
    else:
        arr = preprocess_input(bundle, pd.DataFrame([inputs],columns=columns))
        pred, conf = predict_model(bundle, arr, show_confidence)
        st.write("⚠️ Likely condition detected." if pred else "💚 No condition detected.")
        if conf is not None: st.write(f"Confidence: {np.max(conf):.2f}")
        save_prediction_history(disease_selection, columns, inputs, pred, conf)
        report_text = generate_report(disease_selection, columns, inputs, pred, conf)
        st.download_button("📥 Download Diagnosis Report", report_text, file_name=f"{disease_selection}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# -------------------------
# OCR + NLP REPORT
# -------------------------
st.markdown("---")
st.header("📄 Medical Report Detection (OCR + NLP)")
uploaded_report = st.file_uploader("Upload Report (PDF/Image)", type=["pdf","png","jpg","jpeg"])

if uploaded_report:
    temp_path = Path("temp_uploaded_report")
    temp_path.write_bytes(uploaded_report.read())
    if OCR_AVAILABLE:
        extracted_text = extract_text_from_pdf_util(temp_path)
        if extracted_text:
            st.text_area("📝 Extracted Report Text", extracted_text, height=250)
            if NLP_AVAILABLE and nlp_model:
                try:
                    result = nlp_model(extracted_text[:1000])[0]
                    st.success(f"**NLP Result:** {result['label']} (Confidence: {result['score']:.2f})")
                except Exception as e:
                    st.error(f"NLP failed: {e}")
            else: st.warning("⚠️ NLP model not loaded.")
        else: st.warning("⚠️ No text extracted from report.")
    else: st.warning("⚠️ OCR/NLP libraries missing.")
else:
    st.info("📂 Upload a medical report for analysis.")

st.markdown("---")
st.caption("© 2025 Akshat Sharma | AI Health Assistant v3.5")