# AI Health Assistant v3.5 - Consolidated app.py
# Developer: Akshat Sharma (fixed + hardened)
# Features: Firestore (st.secrets), hybrid OCR (pdf2image+pytesseract fallback -> PyPDF2),
# sklearn + Keras support (safe fallback dummy), IoT simulation + hooks, PDF reports (reportlab).

import os
import io
import json
import random
import time
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np

# ML & utils
import joblib
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

# PDF / OCR tools (optional)
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_BINARIES_AVAILABLE = True
except Exception:
    OCR_BINARIES_AVAILABLE = False

from PyPDF2 import PdfReader

# NLP (optional)
try:
    from transformers import pipeline
    NLP_AVAILABLE = True
except Exception:
    NLP_AVAILABLE = False

# PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle

# Firestore (optional)
db = None
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_SDK_PRESENT = True
except Exception:
    FIREBASE_SDK_PRESENT = False

# ----- CONFIG -----
st.set_page_config(page_title="AI Health Assistant", page_icon="🩺", layout="wide")
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DISEASES = ["Heart", "Diabetes", "Liver", "Blood"]

# ----- FIREBASE SETUP -----
def init_firebase():
    global db
    if not FIREBASE_SDK_PRESENT:
        st.warning("Firebase SDK not installed (pip install firebase-admin) — Firestore disabled.")
        return None
    # 1) Try Streamlit secrets (recommended)
    try:
        firebase_config = st.secrets.get("FIREBASE")
        if firebase_config:
            # If secret is string, parse JSON
            if isinstance(firebase_config, str):
                firebase_config = json.loads(firebase_config)
            cred = credentials.Certificate(dict(firebase_config))
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred)
            db = firestore.client()
            st.success("✅ Firebase connected via Streamlit secrets.")
            return db
    except Exception as e:
        st.warning(f"Using Streamlit secrets for Firebase failed: {e}")
    # 2) Try local service account file path from env var or default
    local_path = os.getenv("FIREBASE_JSON_PATH") or str(BASE_DIR / "AI_Health_UI" / "serviceAccountKey.json")
    if Path(local_path).exists():
        try:
            cred = credentials.Certificate(local_path)
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred)
            db = firestore.client()
            st.success("✅ Firebase connected via local JSON file.")
            return db
        except Exception as e:
            st.error(f"Local Firebase init failed: {e}")
    # final fallback
    st.info("AI_Health_UI/ai-power-structural-health-m-s-firebase-adminsdk-fbsvc-76a80ebaa3.json")
    return None

db = init_firebase()

# ----- HELPERS: models -----
def load_sklearn_joblib(path: Path):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"joblib load failed for {path}: {e}")
        return None

def load_keras_model(path: Path):
    if not KERAS_AVAILABLE:
        return None
    try:
        return keras_load_model(str(path))
    except Exception as e:
        print(f"keras load failed for {path}: {e}")
        return None
def load_model_bundle(disease: str):
    bundle = {}
    model_pkl = MODELS_DIR / f"{disease.lower()}_model.pkl"
    model_h5 = MODELS_DIR / f"{disease.lower()}_model_keras.h5"
    imputer_pkl = MODELS_DIR / f"{disease.lower()}_imputer.pkl"
    scaler_pkl = MODELS_DIR / f"{disease.lower()}_scaler.pkl"

    if model_pkl.exists():
        bundle["model"] = load_sklearn_joblib(model_pkl)
        bundle["type"] = "sklearn"
    elif model_h5.exists():
        model = load_keras_model(model_h5)
        if model:
            bundle["model"] = model
            bundle["type"] = "keras"
        else:
            return None
    else:
        return None

    bundle["imputer"] = load_sklearn_joblib(imputer_pkl) if imputer_pkl.exists() else None
    bundle["scaler"] = load_sklearn_joblib(scaler_pkl) if scaler_pkl.exists() else None
    return bundle

# Provide a safe dummy predictor so UI doesn't crash if models missing
class DummyModel:
    def predict(self, arr):
        # always "no disease" i.e., 0
        return np.array([0 for _ in range(arr.shape[0])])
    def predict_proba(self, arr):
        # low-confidence dummy
        return np.array([[0.9, 0.1] for _ in range(arr.shape[0])])

def safe_bundle_for(disease):
    b = load_model_bundle(disease)
    if b is None:
        # return dummy bundle
        return {"model": DummyModel(), "type": "sklearn", "imputer": None, "scaler": None, "is_dummy": True}
    b["is_dummy"] = False
    return b

model_bundles = {d: safe_bundle_for(d) for d in DISEASES}

# ----- PREPROCESS & PREDICT -----
def preprocess_input(bundle, df_input: pd.DataFrame):
    arr = df_input.values.astype(float)
    imputer, scaler = bundle.get("imputer"), bundle.get("scaler")
    if imputer:
        try:
            arr = imputer.transform(df_input)
        except Exception:
            arr = imputer.transform(pd.DataFrame(arr, columns=df_input.columns))
    if scaler:
        try:
            arr = scaler.transform(arr)
        except Exception:
            arr = scaler.transform(pd.DataFrame(arr, columns=df_input.columns))
    return arr

def predict_model(bundle, arr, show_confidence=True):
    model_type = bundle.get("type", "sklearn")
    model = bundle.get("model")
    conf = None
    if model_type == "sklearn":
        pred = int(model.predict(arr)[0])
        if show_confidence and hasattr(model, "predict_proba"):
            conf = model.predict_proba(arr)[0]
    else:  # keras
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

# ----- FIRESTORE HISTORY -----
def save_prediction_history(disease, columns, input_values, pred, conf):
    record = {
        "Date": datetime.now().isoformat(),
        "Disease": disease,
        "Input": {c: v for c, v in zip(columns, input_values)},
        "Prediction": int(pred),
        "Confidence": float(np.max(conf)) if conf is not None else None
    }
    if db is None:
        st.info("Firebase not connected — saving to local CSV history.")
        hf = BASE_DIR / "prediction_history.csv"
        try:
            df_old = pd.read_csv(hf) if hf.exists() else pd.DataFrame()
            df_new = pd.DataFrame([record])
            pd.concat([df_old, df_new], ignore_index=True).to_csv(hf, index=False)
        except Exception as e:
            st.warning(f"Saving local history failed: {e}")
        return
    try:
        db.collection("predictions").add(record)
    except Exception as e:
        st.warning(f"Failed to save to Firestore: {e}")

# ----- OCR & NLP (hybrid) -----
@st.cache_resource
def load_nlp():
    if not NLP_AVAILABLE:
        return None
    try:
        return pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception as e:
        st.warning(f"NLP pipeline load failed: {e}")
        return None

nlp_model = load_nlp()

from typing import Optional

def extract_text_from_pdf(file_path: Path, poppler_path: Optional[str] = None) -> str:
    """
    Hybrid PDF text extractor:
      - If pdf2image+pytesseract + poppler exist -> use OCR (scanned images)
      - Else fall back to PyPDF2 text extraction
    """
    # OCR route
    if OCR_BINARIES_AVAILABLE:
        try:
            images = convert_from_path(str(file_path), poppler_path=poppler_path) if poppler_path else convert_from_path(str(file_path))
            text = "".join(pytesseract.image_to_string(img) for img in images)
            if text.strip():
                return text.strip()
        except Exception as e:
            st.warning(f"OCR attempt failed, falling back to PyPDF2: {e}")
    # PyPDF2 fallback
    try:
        reader = PdfReader(file_path)
        text = ""
        for p in reader.pages:
            text += p.extract_text() or ""
        return text.strip() or "No extractable text found in the PDF."
    except Exception as e:
        return f"❌ PDF extraction error: {e}"

# ----- PDF report generation (reportlab) -----
def generate_pdf_report(title: str, metadata: dict, table_df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    flow = []
    flow.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    flow.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    flow.append(Paragraph(" ", styles["Normal"]))
    # metadata
    for k, v in metadata.items():
        flow.append(Paragraph(f"<b>{k}:</b> {v}", styles["Normal"]))
    flow.append(Paragraph(" ", styles["Normal"]))
    # table
    data = [table_df.columns.tolist()] + table_df.fillna("").astype(str).values.tolist()
    tbl = Table(data, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2E86AB")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
    ]))
    flow.append(tbl)
    doc.build(flow)
    buffer.seek(0)
    return buffer.read()

# ----- UI -----
st.sidebar.title("AI Health Assistant 🩺 v3.5")
disease_selection = st.sidebar.selectbox("Select Disease Module", DISEASES)
show_confidence = st.sidebar.checkbox("Show Confidence", value=True)
enable_nlp = st.sidebar.checkbox("Enable Report NLP", value=True)
enable_ocr = st.sidebar.checkbox("Enable OCR (local only)", value=False)
enable_iot = st.sidebar.checkbox("Enable Live IoT Input (simulate)", value=False)
poppler_path = st.sidebar.text_input("Poppler bin path (optional, Windows)", value=os.getenv("POPPLER_PATH",""))

st.title(f"🩺 {disease_selection} Diagnosis & Analysis")
st.markdown("---")

st.info({
    "Heart":"Heart: predicts cardiovascular risk from age, bp, chol etc.",
    "Diabetes":"Diabetes: predicts diabetes risk from glucose, BMI, insulin etc.",
    "Liver":"Liver: predicts liver condition from bilrubin, enzymes, albumin etc.",
    "Blood":"Blood: simple RBC/WBC/Hb analysis."
}.get(disease_selection, ""))

# IoT: simulation and placeholders for real integration
def fetch_iot_data_sim(disease):
    if disease == "Heart":
        return {"age": random.randint(20,80), "sex": random.choice([0,1]), "cp": random.randint(0,3),
                "trestbps": random.randint(100,160), "chol": random.randint(150,300), "fbs": random.choice([0,1]),
                "restecg": random.randint(0,2), "thalach": random.randint(90,180),"exang": random.choice([0,1]),
                "oldpeak": round(random.uniform(0,4),1), "slope": random.randint(0,2), "ca": random.randint(0,3), "thal": random.randint(1,3)}
    if disease == "Diabetes":
        return {"Pregnancies": random.randint(0,5), "Glucose": random.randint(70,200), "BloodPressure": random.randint(50,120),
                "SkinThickness": random.randint(10,40), "Insulin": random.randint(10,200), "BMI": round(random.uniform(18,40),1),
                "DiabetesPedigreeFunction": round(random.uniform(0.1,2.0),2), "Age": random.randint(18,80)}
    if disease == "Liver":
        return {"Age": random.randint(18,80), "Gender": random.choice([0,1]), "Total_Bilirubin": round(random.uniform(0.2,3.0),2),
                "Direct_Bilirubin": round(random.uniform(0.1,1.0),2), "Alkaline_Phosphotase": random.randint(40,300),
                "Alamine_Aminotransferase": random.randint(10,200), "Aspartate_Aminotransferase": random.randint(10,200),
                "Total_Protiens": round(random.uniform(5.0,9.0),1), "Albumin": round(random.uniform(2.0,5.0),1),
                "Albumin_and_Globulin_Ratio": round(random.uniform(0.5,2.5),2)}
    if disease == "Blood":
        return {"Hemoglobin": round(random.uniform(9,17),1), "RBC": round(random.uniform(3.0,6.0),2), "WBC": round(random.uniform(3,14),1)}
    return {}

def disease_inputs_ui(disease):
    inputs = []; cols = []
    if disease == "Heart":
        age = st.number_input("Age", 1, 120, 45); sex = st.selectbox("Sex", ["Male","Female"])
        cp = st.selectbox("Chest Pain Type", [0,1,2,3]); trestbps = st.number_input("BP",80,200,130)
        chol = st.number_input("Cholesterol",100,400,200); fbs = st.selectbox("Fasting Sugar>120", [0,1])
        restecg = st.selectbox("ECG Results", [0,1,2]); thalach = st.number_input("Max HR",50,250,150)
        exang = st.selectbox("Exercise Induced Angina",[0,1]); oldpeak = st.number_input("ST Depression",0.0,6.5,1.0,0.1)
        slope = st.selectbox("Slope ST",[0,1,2]); ca = st.number_input("Major Vessels",0,3,0); thal = st.selectbox("Thalassemia",[1,2,3])
        cols = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
        inputs = [age, 1 if sex=="Male" else 0, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    elif disease == "Diabetes":
        pregnancies=st.number_input("Pregnancies",0,20,0); glucose=st.number_input("Glucose",0,300,120)
        bp=st.number_input("BP",0,200,70); skin=st.number_input("Skin Thickness",0,100,20)
        insulin=st.number_input("Insulin",0,900,80); bmi=st.number_input("BMI",0.0,70.0,28.0)
        dpf=st.number_input("Diabetes Pedigree",0.0,2.5,0.5); age=st.number_input("Age",1,120,35)
        cols=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
        inputs=[pregnancies,glucose,bp,skin,insulin,bmi,dpf,age]
    elif disease == "Liver":
        age_v=st.number_input("Age",1,120,45); gender=st.selectbox("Gender",["Male","Female"])
        total_bilirubin=st.number_input("Total Bilirubin",0.0,10.0,1.0); direct_bilirubin=st.number_input("Direct Bilirubin",0.0,5.0,0.2)
        alk_phos=st.number_input("ALP",0,1500,100); sgpt=st.number_input("SGPT",0,1000,30); sgot=st.number_input("SGOT",0,1000,30)
        total_proteins=st.number_input("Total Proteins",0.0,10.0,7.0); albumin=st.number_input("Albumin",0.0,6.0,3.5); ag_ratio=st.number_input("Albumin/Globulin Ratio",0.0,3.0,1.0)
        cols=["Age","Gender","Total_Bilirubin","Direct_Bilirubin","Alkaline_Phosphotase","Alamine_Aminotransferase","Aspartate_Aminotransferase","Total_Protiens","Albumin","Albumin_and_Globulin_Ratio"]
        inputs=[age_v,1 if gender=="Male" else 0,total_bilirubin,direct_bilirubin,alk_phos,sgpt,sgot,total_proteins,albumin,ag_ratio]
    elif disease == "Blood":
        hb=st.number_input("Hemoglobin",0.0,25.0,13.0); rbc=st.number_input("RBC",0.0,10.0,4.5); wbc=st.number_input("WBC",0.0,50.0,6.0)
        cols=["Hemoglobin","RBC","WBC"]; inputs=[hb,rbc,wbc]
    return inputs, cols

if enable_iot:
    iot_data = fetch_iot_data_sim(disease_selection)
    st.info(f"🔴 Live IoT Data (simulated): {iot_data}")
    columns = list(iot_data.keys()); inputs = list(iot_data.values())
else:
    inputs, columns = disease_inputs_ui(disease_selection)

if st.button(f"🔍 Predict {disease_selection}"):
    bundle = model_bundles.get(disease_selection)
    if not bundle:
        st.error(f"{disease_selection} model missing.")
    else:
        try:
            arr = preprocess_input(bundle, pd.DataFrame([inputs], columns=columns))
            pred, conf = predict_model(bundle, arr, show_confidence)
            st.write("⚠️ Likely condition detected." if pred else "💚 No condition detected.")
            if conf is not None:
                st.write(f"Confidence: {np.max(conf):.2f}")
            save_prediction_history(disease_selection, columns, inputs, pred, conf)
            # Build report table
            df_report = pd.DataFrame([inputs], columns=columns)
            metadata = {"Disease": disease_selection, "Prediction": int(pred), "Confidence": float(np.max(conf)) if conf is not None else "N/A"}
            pdf_bytes = generate_pdf_report(f"{disease_selection} Diagnosis Report", metadata, df_report)
            st.download_button("📥 Download Diagnosis PDF", data=pdf_bytes, file_name=f"{disease_selection}_diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("---")
st.header("📄 Medical Report Detection (OCR + NLP)")

uploaded_report = st.file_uploader("Upload Report (PDF/Image)", type=["pdf","png","jpg","jpeg"])
if uploaded_report:
    tmp = Path("temp_uploaded_report.pdf")
    tmp.write_bytes(uploaded_report.read())
    # Use OCR only if user enabled it AND local binaries available
    poppler_path_to_use = poppler_path.strip() or None
    extracted_text = extract_text_from_pdf(tmp, poppler_path=poppler_path_to_use)
    st.text_area("📝 Extracted Report Text", extracted_text, height=250)
    # Save OCR text to PDF report
    df_text = pd.DataFrame({"Extracted Text": [extracted_text]})
    meta = {"Source File": uploaded_report.name}
    pdf_bytes = generate_pdf_report("OCR Extracted Report", meta, df_text)
    st.download_button("📥 Download OCR Report (PDF)", data=pdf_bytes, file_name=f"OCR_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
    # NLP interpretation (if available and enabled)
    if enable_nlp and NLP_AVAILABLE and nlp_model:
        try:
            res = nlp_model(extracted_text[:1000])[0]
            st.success(f"**NLP Result:** {res['label']} (Confidence: {res['score']:.2f})")
        except Exception as e:
            st.error(f"NLP processing failed: {e}")
    else:
        if enable_nlp:
            st.warning("NLP model not available in this environment.")
else:
    st.info("📂 Upload a medical report (PDF/image).")

st.markdown("---")
st.caption("© 2025 Akshat Sharma | AI Health Assistant v3.5 - Hardened")
