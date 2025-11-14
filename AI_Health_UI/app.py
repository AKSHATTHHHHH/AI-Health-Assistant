# app.py — AI Health Assistant v4.0 (Single file, consolidated)
# Developer: Akshat Sharma (improved + hardened)
# Purpose: Realtime IoT -> Firebase -> AI inference -> PDF reports + OCR + AI Doctor
# Run with: streamlit run app.py

import os
import io
import json
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Optional
# ----------------------------
# Optional ML / OCR / NLP imports (safe fallbacks)
# ----------------------------
import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import pipeline

# Firebase Admin
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, db as rtdb
    FIREBASE_SDK_PRESENT = True
except Exception:
    FIREBASE_SDK_PRESENT = False

# joblib (sklearn)
try:
    import joblib
    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False

# Keras/Tensorflow loader safe
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

# OCR: pdf2image + pytesseract
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_IMG_AVAILABLE = True
except Exception:
    OCR_IMG_AVAILABLE = False

# PyPDF2 fallback
try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except Exception:
    PYPDF2_AVAILABLE = False

# Transformers (AI Doctor)
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False
# ReportLab for PDF outputs
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False 

# ----------------------------
# Config / Paths
# ----------------------------
st.set_page_config(page_title="AI Health Assistant v4.0", page_icon="🩺", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
TMP_DIR = BASE_DIR / "tmp"
TMP_DIR.mkdir(exist_ok=True)
DISEASES = ["Heart", "Diabetes", "Liver", "Blood"]

# Firebase config behaviour:
# - Preferred: use Streamlit secrets with key "FIREBASE" containing service account JSON (escaped \n in private_key)
# - Fallback: set env var FIREBASE_JSON_PATH to a local service account JSON file.
# - Choose DB type via env var FIREBASE_DB_TYPE = "firestore" or "realtime" (default: firestore for history, realtime for sensors)
FIREBASE_DB_TYPE = os.getenv("FIREBASE_DB_TYPE", "both").lower()  # "firestore", "realtime", "both"

# Default Realtime DB URL (replace in secrets if different)
DEFAULT_RTD_URL = os.getenv("FIREBASE_RTD_URL", "https://ai-power-structural-health-m-s-default-rtdb.firebaseio.com")

# ----------------------------
# Firebase Initialization (supports Firestore + Realtime) 
# ----------------------------
firebase_firestore_client = None
firebase_rtdb_root = None

def init_firebase():
    global firebase_firestore_client, firebase_rtdb_root
    if not FIREBASE_SDK_PRESENT:
        st.warning("firebase-admin not installed. Firebase disabled. (pip install firebase-admin)")
        return False

    # load service account dict from Streamlit secrets first
    cred_obj = None
    try:
        if "FIREBASE" in st.secrets:
            raw = st.secrets["FIREBASE"]
            if isinstance(raw, str):
                raw = json.loads(raw)
            cred_obj = dict(raw)
    except Exception as e:
        st.warning(f"Failed parsing streamlit secrets FIREBASE: {e}")

    # fallback to file path env var or default path
    if cred_obj is None:
        path = os.getenv("FIREBASE_JSON_PATH") or str(BASE_DIR / "AI_Health_UI"/"ai-power-structural-health-m-s-firebase-adminsdk-fbsvc-08f5f13256.json")
        if Path(path).exists():
            try:
                cred_obj = json.load(open(path, "r"))
            except Exception as e:
                st.warning(f"Failed reading {path}: {e}")
        else:
            # nothing found
            st.info("No Firebase credentials found in Streamlit secrets or FIREBASE_JSON_PATH.")
            return False

    try:
        # create credentials.Certificate expects either dict or path — use dict
        cred = credentials.Certificate(cred_obj)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                # if realtime DB URL present in secrets/env use it
                'databaseURL': DEFAULT_RTD_URL or (cred_obj.get("https://ai-power-structural-health-m-s-default-rtdb.firebaseio.com") if cred_obj else "") or ""
            })
        # init clients
        if FIREBASE_DB_TYPE in ("firestore", "both"):
            firebase_firestore_client = firestore.client()
        if FIREBASE_DB_TYPE in ("realtime", "both"):
            # rtdb.root is available via db.reference('/')
            firebase_rtdb_root = rtdb.reference("/")
        st.success("✅ Firebase initialized.")
        return True
    except Exception as e:
        st.error(f"Firebase init failed: {e}")
        return False

# Attempt initialize
init_firebase()

# ----------------------------
# Model loading & safe dummy
# ----------------------------
class DummyModel:
    def predict(self, X):
        n = X.shape[0]
        return np.array([0]*n)
    def predict_proba(self, X):
        n = X.shape[0]
        return np.array([[0.85, 0.15]]*n)

def safe_load_sklearn(path: Path):
    try:
        if JOBLIB_AVAILABLE:
            return joblib.load(path)
    except Exception as e:
        st.warning(f"Sklearn load failed for {path.name}: {e}")
    return None

def safe_load_keras(path: Path):
    if not KERAS_AVAILABLE:
        return None
    try:
        return keras_load_model(str(path))
    except Exception as e:
        st.warning(f"Keras load failed for {path.name}: {e}")
        return None

def load_bundle_for(disease):
    # search in models dir: {disease_lower}_model.pkl or _model_keras.h5
    d = disease.lower()
    pkl = MODELS_DIR / f"{d}_model.pkl"
    h5 = MODELS_DIR / f"{d}_model_keras.h5"
    imputer = MODELS_DIR / f"{d}_imputer.pkl"
    scaler = MODELS_DIR / f"{d}_scaler.pkl"
    bundle = {"is_dummy": True, "model": DummyModel(), "type": "sklearn", "imputer": None, "scaler": None}
    if pkl.exists():
        m = safe_load_sklearn(pkl)
        if m: bundle.update({"is_dummy": False, "model": m, "type": "sklearn"})
    elif h5.exists():
        m = safe_load_keras(h5)
        if m: bundle.update({"is_dummy": False, "model": m, "type": "keras"})
    if Path(imputer).exists() and JOBLIB_AVAILABLE:
        try:
            bundle["imputer"] = joblib.load(imputer)
        except Exception:
            bundle["imputer"] = None
    if Path(scaler).exists() and JOBLIB_AVAILABLE:
        try:
            bundle["scaler"] = joblib.load(scaler)
        except Exception:
            bundle["scaler"] = None
    return bundle

model_bundles = {d: load_bundle_for(d) for d in DISEASES}

def preprocess(bundle, df: pd.DataFrame):
    arr = df.values.astype(float)
    imputer = bundle.get("imputer")
    scaler = bundle.get("scaler")
    if imputer is not None:
        try:
            arr = imputer.transform(df)
        except Exception:
            try:
                arr = imputer.transform(pd.DataFrame(arr, columns=df.columns))
            except Exception:
                pass
    if scaler is not None:
        try:
            arr = scaler.transform(arr)
        except Exception:
            try:
                arr = scaler.transform(pd.DataFrame(arr, columns=df.columns))
            except Exception:
                pass
    return arr

def predict(bundle, arr):
    try:
        if bundle.get("type") == "sklearn":
            model = bundle["model"]
            pred = int(model.predict(arr)[0])
            conf = None
            if hasattr(model, "predict_proba"):
                try:
                    conf = model.predict_proba(arr)[0]
                except Exception:
                    conf = None
            return pred, conf
        else:
            model = bundle["model"]
            probs = model.predict(arr)
            if probs.ndim == 2 and probs.shape[1] > 1:
                pred = int(np.argmax(probs[0]))
                conf = probs[0]
            elif probs.ndim == 2 and probs.shape[1] == 1:
                p = float(probs[0][0]); pred = 1 if p >= 0.5 else 0; conf = np.array([1-p, p])
            else:
                pred = int(np.round(float(probs[0][0])))
                conf = None
            return pred, conf
    except Exception as e:
        st.warning(f"Prediction error: {e}")
        return 0, None

# ----------------------------
# Firebase write helpers
# ----------------------------
def save_prediction_to_firestore(record: dict):
    if not FIREBASE_SDK_PRESENT or firebase_firestore_client is None:
        return False
    try:
        firebase_firestore_client.collection("predictions").add(record)
        return True
    except Exception as e:
        st.warning(f"Failed to save prediction to Firestore: {e}")
        return False

def push_iot_to_realtime(path: str, payload: dict):
    # path e.g. "iot/heart/sensor1"
    if not FIREBASE_SDK_PRESENT or firebase_rtdb_root is None:
        return False
    try:
        firebase_rtdb_root.child(path).push(json.dumps(payload))
        return True
    except Exception as e:
        st.warning(f"Failed to push to Realtime DB: {e}")
        return False

# ----------------------------
# OCR utilities
# ----------------------------
def extract_text_from_pdf(file_path: Path, poppler_path: Optional[str] = None) -> str:
    # Try image OCR first if available (handles scanned PDFs)
    if OCR_IMG_AVAILABLE:
        try:
            # convert_from_path accepts poppler_path keyword on Windows
            imgs = convert_from_path(str(file_path), poppler_path=poppler_path) if poppler_path else convert_from_path(str(file_path))
            text_parts = []
            for img in imgs:
                try:
                    t = pytesseract.image_to_string(img)
                    text_parts.append(t)
                except Exception as e:
                    # image OCR failed for page
                    continue
            text = "\n".join(text_parts).strip()
            if text:
                return text
        except Exception as e:
            # fallback to PyPDF2 below
            st.warning(f"OCR (pdf2image+pytesseract) failed, falling back: {e}")

    if PYPDF2_AVAILABLE:
        try:
            reader = PdfReader(str(file_path))
            txt = []
            for p in reader.pages:
                try:
                    txt.append(p.extract_text() or "")
                except Exception:
                    continue
            combined = "\n".join(txt).strip()
            if combined:
                return combined
            else:
                return "No extractable text found."
        except Exception as e:
            return f"PDF extraction error: {e}"
    return "No OCR engines available."

# ----------------------------
# PDF report generation (ReportLab) — readable table format
# ----------------------------
def generate_pdf(title: str, metadata: dict, df_table: pd.DataFrame) -> bytes:
    if not REPORTLAB_AVAILABLE:
        # fallback: return simple text-PDF via bytes
        text = f"{title}\n\nMetadata:\n" + "\n".join([f"{k}: {v}" for k, v in metadata.items()]) + "\n\nTable:\n" + df_table.to_string()
        return text.encode("utf-8")
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=20, leftMargin=20, topMargin=30, bottomMargin=18)
    styles = getSampleStyleSheet()
    flow = []
    flow.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    flow.append(Spacer(1, 6))
    flow.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    flow.append(Spacer(1, 6))
    # metadata
    for k, v in metadata.items():
        flow.append(Paragraph(f"<b>{k}:</b> {v}", styles["Normal"]))
    flow.append(Spacer(1, 12))
    # table data
    data = [df_table.columns.tolist()] + df_table.fillna("").astype(str).values.tolist()
    tbl = Table(data, hAlign="LEFT", repeatRows=1)
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2E86AB")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    flow.append(tbl)
    doc.build(flow)
    buf.seek(0)
    return buf.read()

# ----------------------------
# AI Doctor (summarize + suggestions)
# ----------------------------
@st.cache_resource
def load_ai_doctor():
    if HF_AVAILABLE:
        try:
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
            return {"summarizer": summarizer, "classifier": classifier}
        except Exception as e:
            st.warning(f"Transformers pipelines failed to load: {e}")
            return None
    return None

ai_doctor = load_ai_doctor()

def ai_doctor_analysis(text, prediction_summary=None):
    if not HF_AVAILABLE:
        return {
            "summary": "Advanced AI model not available on cloud.",
            "recommendations": "Use OpenAI API or local GPU.",
            "severity": "N/A",
            "explain": "transformers is not installed."
        }

    # If transformers present, use them for better summaries
    if ai_doctor is not None:
        try:
            summ = ai_doctor["summarizer"](text[:6000], max_length=120, min_length=40, truncation=True)[0]['summary_text']
            cls = ai_doctor["classifier"](text[:1000])[0]
            sentiment = cls.get("label", "NEUTRAL")
            score = float(cls.get("score", 0.0))
            severity = "High" if "NEG" in sentiment and score > 0.8 else ("Medium" if score > 0.6 else "Low")
            recommendations = "Recommend immediate clinical follow-up." if severity == "High" else ("Consider lifestyle changes and follow-up tests." if severity == "Medium" else "Low risk — routine monitoring.")
            explain = f"Sentiment: {sentiment} (score {score:.2f})."
            if prediction_summary:
                explain = f"{explain} Model prediction: {prediction_summary}."
            return {"summary": summ, "recommendations": recommendations, "severity": severity, "explain": explain}
        except Exception as e:
            st.warning(f"AI Doctor pipeline failure: {e}")
    # fallback rule-based
    lower = text.lower()
    severity = "Low"
    if any(word in lower for word in ["critical", "tumor", "infarct", "stroke", "cancer", "severe", "high risk", "emergency"]):
        severity = "High"
    elif any(word in lower for word in ["elevated", "abnormal", "borderline", "concern"]):
        severity = "Medium"
    summary = (text[:500] + "...") if len(text) > 500 else text
    recommendations = "See a clinician immediately." if severity == "High" else ("Review with doctor and run follow-up tests." if severity == "Medium" else "Routine monitoring recommended.")
    explain = "Fallback analysis (transformers not available)."
    if prediction_summary:
        explain += f" Model prediction: {prediction_summary}."
    return {"summary": summary, "recommendations": recommendations, "severity": severity, "explain": explain}

# ----------------------------
# UI: Sidebar Controls
# ----------------------------
st.sidebar.title("AI Health Assistant 🩺 v4.0")
disease_selection = st.sidebar.selectbox("Select Disease Module", DISEASES)
show_confidence = st.sidebar.checkbox("Show Confidence", value=True)
enable_nlp = st.sidebar.checkbox("Enable AI Doctor (NLP)", value=True)
enable_ocr = st.sidebar.checkbox("Enable OCR (local only)", value=True)
enable_iot = st.sidebar.checkbox("Enable Live IoT Input (simulate)", value=False)
poppler_path = st.sidebar.text_input("Poppler bin path (Windows) (optional)", value=os.getenv("POPPLER_PATH",""))
st.sidebar.markdown("---")
st.sidebar.markdown("**Firebase**: Make sure `FIREBASE` secret or `FIREBASE_JSON_PATH` is configured.")
st.title(f"🩺 {disease_selection} Diagnosis & Analysis")

# ----------------------------
# IoT simulation
# ----------------------------
def fetch_iot_sim(disease):
    if disease == "Heart":
        return {"age": random.randint(20,80), "sex": random.choice([0,1]), "cp": random.randint(0,3), "trestbps": random.randint(100,160),
                "chol": random.randint(150,300), "fbs": random.choice([0,1]), "restecg": random.randint(0,2), "thalach": random.randint(90,180),
                "exang": random.choice([0,1]), "oldpeak": round(random.uniform(0,4),1), "slope": random.randint(0,2), "ca": random.randint(0,3), "thal": random.randint(1,3)}
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

# Inputs UI builder
def build_inputs_ui(disease):
    if disease == "Heart":
        age = st.number_input("Age", 1, 120, 45); sex = st.selectbox("Sex", ["Male","Female"])
        cp = st.selectbox("Chest Pain Type", [0,1,2,3]); trestbps = st.number_input("BP",80,200,130)
        chol = st.number_input("Cholesterol",100,400,200); fbs = st.selectbox("Fasting Sugar>120", [0,1])
        restecg = st.selectbox("ECG Results", [0,1,2]); thalach = st.number_input("Max HR",50,250,150)
        exang = st.selectbox("Exercise Induced Angina",[0,1]); oldpeak = st.number_input("ST Depression",0.0,6.5,1.0,0.1)
        slope = st.selectbox("Slope ST",[0,1,2]); ca = st.number_input("Major Vessels",0,3,0); thal = st.selectbox("Thalassemia",[1,2,3])
        cols = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
        vals = [age, 1 if sex=="Male" else 0, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        return vals, cols
    if disease == "Diabetes":
        pregnancies=st.number_input("Pregnancies",0,20,0); glucose=st.number_input("Glucose",0,300,120)
        bp=st.number_input("BP",0,200,70); skin=st.number_input("Skin Thickness",0,100,20)
        insulin=st.number_input("Insulin",0,900,80); bmi=st.number_input("BMI",0.0,70.0,28.0)
        dpf=st.number_input("Diabetes Pedigree",0.0,2.5,0.5); age=st.number_input("Age",1,120,35)
        cols=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
        vals=[pregnancies,glucose,bp,skin,insulin,bmi,dpf,age]
        return vals, cols
    if disease == "Liver":
        age_v=st.number_input("Age",1,120,45); gender=st.selectbox("Gender",["Male","Female"])
        total_bilirubin=st.number_input("Total Bilirubin",0.0,10.0,1.0); direct_bilirubin=st.number_input("Direct Bilirubin",0.0,5.0,0.2)
        alk_phos=st.number_input("ALP",0,1500,100); sgpt=st.number_input("SGPT",0,1000,30); sgot=st.number_input("SGOT",0,1000,30)
        total_proteins=st.number_input("Total Proteins",0.0,10.0,7.0); albumin=st.number_input("Albumin",0.0,6.0,3.5); ag_ratio=st.number_input("Albumin/Globulin Ratio",0.0,3.0,1.0)
        cols=["Age","Gender","Total_Bilirubin","Direct_Bilirubin","Alkaline_Phosphotase","Alamine_Aminotransferase","Aspartate_Aminotransferase","Total_Protiens","Albumin","Albumin_and_Globulin_Ratio"]
        vals=[age_v,1 if gender=="Male" else 0,total_bilirubin,direct_bilirubin,alk_phos,sgpt,sgot,total_proteins,albumin,ag_ratio]
        return vals, cols
    if disease == "Blood":
        hb=st.number_input("Hemoglobin",0.0,25.0,13.0); rbc=st.number_input("RBC",0.0,10.0,4.5); wbc=st.number_input("WBC",0.0,50.0,6.0)
        cols=["Hemoglobin","RBC","WBC"]; vals=[hb,rbc,wbc]
        return vals, cols
    return [], []

# choose input source
if enable_iot:
    iot_payload = fetch_iot_sim(disease_selection)
    st.info(f"🔴 Live IoT Data (simulated): {iot_payload}")
    inputs = list(iot_payload.values()); columns = list(iot_payload.keys())
    # push to realtime if configured
    if firebase_rtdb_root is not None:
        push_iot_to_realtime(f"iot/{disease_selection.lower()}/sim", {"timestamp": datetime.utcnow().isoformat(), "payload": iot_payload})
else:
    inputs, columns = build_inputs_ui(disease_selection)

# Prediction action
if st.button(f"🔍 Predict {disease_selection}"):
    bundle = model_bundles.get(disease_selection, {"model": DummyModel(), "type": "sklearn", "imputer": None, "scaler": None})
    try:
        df_in = pd.DataFrame([inputs], columns=columns)
        arr = preprocess(bundle, df_in)
        pred, conf = predict(bundle, arr)
        human = "Likely condition detected" if pred else "No condition detected"
        st.success(human if pred else "💚 No condition detected.")
        if conf is not None and show_confidence:
            st.write(f"Confidence: {np.max(conf):.2f}")
        # Save record
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "disease": disease_selection,
            "input_columns": columns,
            "input_values": inputs,
            "prediction": int(pred),
            "confidence": float(np.max(conf)) if conf is not None else None,
            "is_dummy_model": bundle.get("is_dummy", True)
        }
        # write to firestore
        if firebase_firestore_client is not None:
            try:
                firebase_firestore_client.collection("predictions").add(record)
            except Exception as e:
                st.warning(f"Failed Firestore write: {e}")
        else:
            # local fallback: append csv
            csvf = BASE_DIR / "local_prediction_history.csv"
            try:
                dfold = pd.read_csv(csvf) if csvf.exists() else pd.DataFrame()
                dfnew = pd.DataFrame([{
                    "timestamp": record["timestamp"],
                    "disease": record["disease"],
                    "prediction": record["prediction"],
                    "confidence": record["confidence"]
                }])
                pd.concat([dfold, dfnew], ignore_index=True).to_csv(csvf, index=False)
            except Exception as e:
                st.warning(f"Failed local persistence: {e}")

        # AI Doctor analysis
        diagnosis_summary = f"{disease_selection} prediction = {pred}"
        ocr_text_for_ai = ""
        ai_result = {}
        if enable_nlp:
            # basic prompt: combine features + ocr_text if present (we'll pass small text)
            combined = " ".join([f"{c}:{v}" for c, v in zip(columns, inputs)]) + ". " + (ocr_text_for_ai[:2000] if ocr_text_for_ai else "")
            ai_result = ai_doctor_analysis(combined, prediction_summary=diagnosis_summary)

        # PDF report
        df_report = df_in.copy()
        metadata = {"Disease": disease_selection, "Prediction": int(pred), "Confidence": float(np.max(conf)) if conf is not None else "N/A", "Analyzed At": datetime.utcnow().isoformat()}
        if ai_result:
            metadata["AI_Summary"] = ai_result.get("summary", "")[:200]
            metadata["AI_Severity"] = ai_result.get("severity", "")
        pdf_bytes = generate_pdf(f"{disease_selection} Diagnosis Report", metadata, df_report)
        st.download_button("📥 Download Diagnosis PDF", data=pdf_bytes, file_name=f"{disease_selection}_diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ----------------------------
# OCR + AI Doctor for uploaded reports
# ----------------------------
st.markdown("---")
st.header("📄 Medical Report Detection (OCR + AI Doctor)")

uploaded = st.file_uploader("Upload Report (PDF / Image)", type=["pdf", "png", "jpg", "jpeg"])

if uploaded is not None:
    try:
        tmp_path = TMP_DIR / f"uploaded_{int(time.time())}.pdf"
        content = uploaded.read()
        ext = uploaded.name.split(".")[-1].lower()

        # Handle image vs PDF
        if ext in ("png", "jpg", "jpeg"):
            img_path = TMP_DIR / f"img_{int(time.time())}.{ext}"
            with open(img_path, "wb") as f:
                f.write(content)
            if OCR_IMG_AVAILABLE:
                from PIL import Image
                text = pytesseract.image_to_string(Image.open(img_path))
            else:
                text = "Image OCR not available (install pdf2image+pytesseract)."
        else:
            tmp_path.write_bytes(content)
            text = extract_text_from_pdf(tmp_path, poppler_path=poppler_path.strip() or None)

        st.text_area("📝 Extracted Report Text", text[:200000], height=300)

        # --- Firebase or local save ---
        rec = {
            "timestamp": datetime.utcnow().isoformat(),
            "source_file": uploaded.name,
            "extracted_text": text[:10000]
        }

        if firebase_firestore_client is not None:
            try:
                firebase_firestore_client.collection("ocr_reports").add(rec)
            except Exception as e:
                st.warning(f"Failed to save OCR to Firestore: {e}")
        else:
            try:
                csvp = BASE_DIR / "local_ocr_reports.csv"
                dfold = pd.read_csv(csvp) if csvp.exists() else pd.DataFrame()
                dfnew = pd.DataFrame([{
                    "timestamp": rec["timestamp"],
                    "source_file": rec["source_file"],
                    "text": rec["extracted_text"][:300]
                }])
                pd.concat([dfold, dfnew], ignore_index=True).to_csv(csvp, index=False)
            except Exception as e:
                st.warning(f"Failed local OCR persistence: {e}")

        # --- AI Doctor Section ---
        if enable_nlp:
            try:
                ai = ai_doctor_analysis(text, prediction_summary=None)
                st.subheader("AI Doctor Analysis")
                st.write("**Summary:**", ai.get("summary", "N/A"))
                st.write("**Recommendations:**", ai.get("recommendations", "N/A"))
                st.write("**Severity:**", ai.get("severity", "N/A"))
                st.write("**Explanation:**", ai.get("explain", "N/A"))
                meta = {"Source": uploaded.name, "AI_Summary": ai.get("summary", ""), "AI_Severity": ai.get("severity", "")}
            except Exception as e:
                st.warning(f"AI Doctor failed: {e}")
                meta = {"Source": uploaded.name}
        else:
            meta = {"Source": uploaded.name}

        # --- Safe PDF Generation (Split long text) ---
        from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet
        from io import BytesIO

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()
        story = []

        max_chars = 2500
        for i in range(0, len(text), max_chars):
            story.append(Paragraph(text[i:i + max_chars], styles["Normal"]))
            story.append(PageBreak())

        doc.build(story)
        pdf_bytes = buffer.getvalue()

        st.download_button(
            "📥 Download OCR Report (PDF)",
            data=pdf_bytes,
            file_name=f"OCR_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )

    except Exception as e:
        st.error(f"OCR processing failed: {e}")

else:
    st.info("📂 Upload a medical report (PDF/image) to extract and analyze.")

st.markdown("---")
st.caption("© 2025 Akshat Sharma | AI Health Assistant v4.0")