# app.py — AI Health Assistant v4.1
# Developer: Akshat Sharma (improved + hardened)
# Purpose: Realtime IoT -> Firebase -> AI inference -> PDF reports + OCR + AI Doctor + 3D Chatbot
# Run with: streamlit run app.py

import os
import io
import json
import time
import random
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

# UI + data
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Optional heavy libs — import safely
try:
    import joblib
    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False

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

# OCR
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

# Firebase Admin
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, db as rtdb
    FIREBASE_SDK_PRESENT = True
except Exception:
    FIREBASE_SDK_PRESENT = False

# Transformers AI Doctor
try:
    from transformers import (
        pipeline,
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
    )
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# ReportLab for PDF outputs
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# ----------------------------
# Basic App config & paths
# ----------------------------
st.set_page_config(page_title="AI Health Assistant v4.1", page_icon="🩺", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
TMP_DIR = BASE_DIR / "tmp"
TMP_DIR.mkdir(exist_ok=True)
AI_DOCTOR_LOCAL = BASE_DIR / "ai_doctor_models"
AI_DOCTOR_LOCAL.mkdir(exist_ok=True)
AI_SUMMARIZER_LOCAL = AI_DOCTOR_LOCAL / "summarizer"
AI_CLASSIFIER_LOCAL = AI_DOCTOR_LOCAL / "classifier"
DISEASES = ["Heart", "Diabetes", "Liver", "Blood"]

# Local DB fallback (JSON)
LOCAL_DB = BASE_DIR / "local_db.json"
if not LOCAL_DB.exists():
    LOCAL_DB.write_text(json.dumps({"predictions": [], "ocr_reports": []}, indent=2))

# Firebase config behaviour
FIREBASE_DB_TYPE = os.getenv("FIREBASE_DB_TYPE", "both").lower()  # "firestore", "realtime", "both"
DEFAULT_RTD_URL = os.getenv("FIREBASE_RTD_URL", "https://ai-power-structural-health-m-s-default-rtdb.firebaseio.com")

# ----------------------------
# Feature toggles (UI-driven)
# ----------------------------
st.sidebar.title("AI Health Assistant Controls")
MODE = st.sidebar.selectbox("Mode", ["Online", "Offline"], index=0)
IS_ONLINE = MODE == "Online"
ENABLE_FIREBASE = st.sidebar.checkbox("Enable Firebase (if configured)", value=True)
ENABLE_OCR = st.sidebar.checkbox("Enable OCR (pdf2image+pytesseract)", value=True)
ENABLE_NLP = st.sidebar.checkbox("Enable AI Doctor (NLP)", value=True)
ENABLE_IOT_SIM = st.sidebar.checkbox("Enable IoT simulation (push data)", value=False)
SHOW_3D_CHATBOT = st.sidebar.checkbox("Show 3D Chatbot", value=True)
POPPLER_PATH = st.sidebar.text_input("Poppler path (Windows)", value=os.getenv("POPPLER_PATH", ""))
st.sidebar.markdown("---")
st.sidebar.markdown("Make sure to configure `FIREBASE` secret or set `FIREBASE_JSON_PATH` env if using Firebase.")

# ----------------------------
# Utility helpers
# ----------------------------
def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except Exception:
        pass

def now_iso():
    return datetime.utcnow().isoformat()

# ----------------------------
# Firebase initialization
# ----------------------------
firebase_firestore_client = None
firebase_rtdb_root = None

def init_firebase():
    global firebase_firestore_client, firebase_rtdb_root
    if not FIREBASE_SDK_PRESENT or not ENABLE_FIREBASE or not IS_ONLINE:
        safe_print("Firebase not initialized (sdk installed:", FIREBASE_SDK_PRESENT, "enable:", ENABLE_FIREBASE, "online:", IS_ONLINE, ")")
        return False

    cred_obj = None
    try:
        if "FIREBASE" in st.secrets:
            raw = st.secrets["FIREBASE"]
            if isinstance(raw, str):
                raw = json.loads(raw)
            cred_obj = dict(raw)
    except Exception as e:
        st.warning(f"Failed parsing Streamlit secrets FIREBASE: {e}")

    if cred_obj is None:
        path = os.getenv("FIREBASE_JSON_PATH") or str(BASE_DIR / "ai-power-structural-health-m-s-firebase-adminsdk.json")
        if Path(path).exists():
            try:
                cred_obj = json.load(open(path, "r"))
            except Exception as e:
                st.warning(f"Failed reading {path}: {e}")
        else:
            st.info("No Firebase credentials found (streamlit secrets or FIREBASE_JSON_PATH).")
            return False

    try:
        cred = credentials.Certificate(cred_obj)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                'databaseURL': os.getenv("FIREBASE_RTD_URL", DEFAULT_RTD_URL)
            })
        if FIREBASE_DB_TYPE in ("firestore", "both"):
            firebase_firestore_client = firestore.client()
        if FIREBASE_DB_TYPE in ("realtime", "both"):
            firebase_rtdb_root = rtdb.reference("/")
        st.success("✅ Firebase initialized.")
        return True
    except Exception as e:
        st.error(f"Firebase init failed: {e}")
        safe_print(traceback.format_exc())
        return False

# Only attempt init if requested
if ENABLE_FIREBASE and IS_ONLINE:
    init_firebase()

# ----------------------------
# Local DB helpers (offline fallback)
# ----------------------------
def read_local_db() -> dict:
    try:
        return json.loads(LOCAL_DB.read_text())
    except Exception:
        return {"predictions": [], "ocr_reports": []}

def write_local_db(data: dict) -> bool:
    try:
        LOCAL_DB.write_text(json.dumps(data, indent=2))
        return True
    except Exception:
        return False

def save_prediction_offline(record: dict) -> bool:
    d = read_local_db()
    d["predictions"].append(record)
    return write_local_db(d)

def save_ocr_offline(record: dict) -> bool:
    d = read_local_db()
    d["ocr_reports"].append(record)
    return write_local_db(d)

def sync_offline_to_firebase(limit: int = 500) -> Tuple[int, int]:
    """Uploads offline records to Firestore. Returns (pred_uploaded, ocr_uploaded)."""
    if not FIREBASE_SDK_PRESENT or firebase_firestore_client is None:
        return 0, 0
    d = read_local_db()
    p_count = 0
    o_count = 0
    new_preds = []
    new_ocrs = []
    for rec in d.get("predictions", [])[:limit]:
        try:
            firebase_firestore_client.collection("predictions").add(rec)
            p_count += 1
        except Exception:
            new_preds.append(rec)
    for rec in d.get("ocr_reports", [])[:limit]:
        try:
            firebase_firestore_client.collection("ocr_reports").add(rec)
            o_count += 1
        except Exception:
            new_ocrs.append(rec)
    # keep remaining unsynced
    d["predictions"] = new_preds + d.get("predictions", [])[limit:]
    d["ocr_reports"] = new_ocrs + d.get("ocr_reports", [])[limit:]
    write_local_db(d)
    return p_count, o_count

# ----------------------------
# Model bundles loading (sklearn / keras / dummy)
# ----------------------------
class DummyModel:
    def predict(self, X):
        n = X.shape[0]
        return np.array([0]*n)
    def predict_proba(self, X):
        n = X.shape[0]
        return np.array([[0.6, 0.4]]*n)

def safe_load_sklearn(path: Path):
    if not JOBLIB_AVAILABLE:
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Sklearn load failed for {path.name}: {e}")
        return None

def safe_load_keras_model(path: Path):
    if not KERAS_AVAILABLE:
        return None
    try:
        return keras_load_model(str(path))
    except Exception as e:
        st.warning(f"Keras load failed for {path.name}: {e}")
        return None

def load_bundle_for(disease: str) -> dict:
    d = disease.lower()
    pkl = MODELS_DIR / f"{d}_model.pkl"
    h5 = MODELS_DIR / f"{d}_model_keras.h5"
    imputer = MODELS_DIR / f"{d}_imputer.pkl"
    scaler = MODELS_DIR / f"{d}_scaler.pkl"
    bundle = {"is_dummy": True, "model": DummyModel(), "type": "sklearn", "imputer": None, "scaler": None}
    if pkl.exists() and JOBLIB_AVAILABLE:
        m = safe_load_sklearn(pkl)
        if m:
            bundle.update({"is_dummy": False, "model": m, "type": "sklearn"})
    elif h5.exists() and KERAS_AVAILABLE:
        m = safe_load_keras_model(h5)
        if m:
            bundle.update({"is_dummy": False, "model": m, "type": "keras"})
    if imputer.exists() and JOBLIB_AVAILABLE:
        try:
            bundle["imputer"] = joblib.load(imputer)
        except Exception:
            bundle["imputer"] = None
    if scaler.exists() and JOBLIB_AVAILABLE:
        try:
            bundle["scaler"] = joblib.load(scaler)
        except Exception:
            bundle["scaler"] = None
    return bundle

model_bundles = {d: load_bundle_for(d) for d in DISEASES}

def preprocess(bundle: dict, df: pd.DataFrame) -> np.ndarray:
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

def predict(bundle: dict, arr: np.ndarray) -> Tuple[int, Optional[np.ndarray]]:
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
            if hasattr(probs, "ndim") and probs.ndim == 2:
                if probs.shape[1] > 1:
                    pred = int(np.argmax(probs[0]))
                    conf = probs[0]
                elif probs.shape[1] == 1:
                    p = float(probs[0][0])
                    pred = 1 if p >= 0.5 else 0
                    conf = np.array([1-p, p])
                else:
                    pred = int(np.round(float(probs[0][0])))
                    conf = None
            else:
                pred = int(np.round(float(probs[0])))
                conf = None
            return pred, conf
    except Exception as e:
        st.warning(f"Prediction error: {e}")
        return 0, None

# ----------------------------
# Firebase write helpers (robust)
# ----------------------------
def save_prediction_to_firestore(record: dict) -> bool:
    if not FIREBASE_SDK_PRESENT or firebase_firestore_client is None or not IS_ONLINE:
        return False
    try:
        firebase_firestore_client.collection("predictions").add(record)
        return True
    except Exception as e:
        st.warning(f"Failed to save prediction to Firestore: {e}")
        return False

def push_iot_to_realtime(path: str, payload: dict) -> bool:
    if not FIREBASE_SDK_PRESENT or firebase_rtdb_root is None or not IS_ONLINE:
        return False
    try:
        firebase_rtdb_root.child(path).set(payload)
        return True
    except Exception as e:
        st.warning(f"Failed to push to Realtime DB: {e}")
        return False

# ----------------------------
# OCR utilities
# ----------------------------
if OCR_IMG_AVAILABLE and ENABLE_OCR:
    # default tesseract path for windows — change if needed
    try:
        if os.name == "nt":
            if pytesseract.pytesseract.tesseract_cmd is None:
                pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
    except Exception:
        pass

def extract_text_from_pdf(file_path: Path, poppler_path: Optional[str] = None) -> str:
    # Try image OCR first (handles scanned PDFs)
    if OCR_IMG_AVAILABLE and ENABLE_OCR:
        try:
            imgs = convert_from_path(str(file_path), poppler_path=poppler_path) if poppler_path else convert_from_path(str(file_path))
            text_parts = []
            for img in imgs:
                try:
                    t = pytesseract.image_to_string(img)
                    text_parts.append(t)
                except Exception:
                    continue
            text = "\n".join(text_parts).strip()
            if text:
                return text
        except Exception as e:
            st.warning(f"OCR (pdf2image+pytesseract) failed, falling back to PyPDF2: {e}")

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
# PDF report generation
# ----------------------------
def generate_pdf(title: str, metadata: dict, df_table: pd.DataFrame) -> bytes:
    if not REPORTLAB_AVAILABLE:
        # fallback: simple text -> bytes
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
    for k, v in metadata.items():
        flow.append(Paragraph(f"<b>{k}:</b> {v}", styles["Normal"]))
    flow.append(Spacer(1, 12))
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
# AI Doctor: dual-mode (offline local models first, online fallback)
# ----------------------------
@st.cache_resource
def load_ai_pipelines():
    """Load summarizer and classifier pipelines. Try local first, else online. Return dict or None."""
    summarizer = None
    classifier = None
    # Try local summarizer
    try:
        if AI_SUMMARIZER_LOCAL.exists() and HF_AVAILABLE:
            safe_print("Using local summarizer model.")
            tok = AutoTokenizer.from_pretrained(str(AI_SUMMARIZER_LOCAL))
            model = AutoModelForSeq2SeqLM.from_pretrained(str(AI_SUMMARIZER_LOCAL))
            summarizer = pipeline("summarization", model=model, tokenizer=tok)
        elif HF_AVAILABLE:
            safe_print("Using online summarizer model.")
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        safe_print("Summarizer load failed:", e)

    # Try local classifier
    try:
        if AI_CLASSIFIER_LOCAL.exists() and HF_AVAILABLE:
            safe_print("Using local classifier model.")
            tok = AutoTokenizer.from_pretrained(str(AI_CLASSIFIER_LOCAL))
            model = AutoModelForSequenceClassification.from_pretrained(str(AI_CLASSIFIER_LOCAL))
            classifier = pipeline("text-classification", model=model, tokenizer=tok)
        elif HF_AVAILABLE:
            safe_print("Using online classifier model.")
            classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception as e:
        safe_print("Classifier load failed:", e)

    if summarizer is None and classifier is None:
        return None
    return {"summarizer": summarizer, "classifier": classifier}

AI_PIPELINES = load_ai_pipelines()

def ai_doctor_analysis(text: str, prediction_summary: Optional[str] = None) -> Dict[str,str]:
    """Return summary, recommendations, severity, explain."""
    if not ENABLE_NLP:
        return {"summary": "NLP disabled", "recommendations": "", "severity": "N/A", "explain": "NLP disabled in settings."}

    # If pipelines loaded (online or local)
    if AI_PIPELINES is not None:
        try:
            summ_pipe = AI_PIPELINES.get("summarizer")
            cls_pipe = AI_PIPELINES.get("classifier")
            summ = summ_pipe(text[:6000], max_length=120, min_length=40, truncation=True)[0]['summary_text'] if summ_pipe else text[:300]
            cls = cls_pipe(text[:1000])[0] if cls_pipe else {"label": "NEUTRAL", "score": 0.5}
            sentiment = cls.get("label", "NEUTRAL")
            score = float(cls.get("score", 0.0))
            severity = "High" if ("NEG" in sentiment or "NEGATIVE" in sentiment) and score > 0.8 else ("Medium" if score > 0.6 else "Low")
            recommendations = "Recommend immediate clinical follow-up." if severity == "High" else ("Consider lifestyle changes and follow-up tests." if severity == "Medium" else "Low risk — routine monitoring.")
            explain = f"Sentiment: {sentiment} (score {score:.2f})."
            if prediction_summary:
                explain = f"{explain} Model prediction: {prediction_summary}."
            return {"summary": summ, "recommendations": recommendations, "severity": severity, "explain": explain}
        except Exception as e:
            safe_print("AI doctor pipeline error:", e)
            # fallthrough to rule-based

    # Rule-based fallback
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
# UI: Sidebar quick buttons
# ----------------------------
st.sidebar.markdown("---")
if IS_ONLINE and ENABLE_FIREBASE and FIREBASE_SDK_PRESENT and st.sidebar.button("Sync offline -> Firebase"):
    p,u = sync_offline_to_firebase()
    st.sidebar.success(f"Synced {p} predictions and {u} OCRs to Firebase.")

if st.sidebar.button("Show Local DB (raw)"):
    st.sidebar.code(LOCAL_DB.read_text()[:2000] + ("..." if LOCAL_DB.stat().st_size>2000 else ""))

st.title("🩺 AI Health Assistant v4.1")

# ----------------------------
# Inputs UI functions
# ----------------------------
def fetch_iot_sim(disease: str) -> dict:
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

def build_inputs_ui(disease: str):
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

# ----------------------------
# 3D Chatbot (embedded Three.js + chat UI)
# ----------------------------
def render_3d_chatbot():
    # Minimal three.js scene with a rotating head-like sphere and text UI below for chat
    html = f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <style>
          body {{ margin:0; overflow:hidden; background: transparent; }}
          #chatbox {{ font-family: Arial; position: absolute; bottom: 0; left: 12px; width: 95%; max-width: 980px; }}
          .bubble {{ background: rgba(255,255,255,0.9); padding: 8px 12px; border-radius: 10px; margin:6px 0; width: fit-content; max-width: 90%; }}
          .user {{ background: rgba(38,162,255,0.95); color: white; margin-left:auto; }}
        </style>
      </head>
      <body>
        <canvas id="c"></canvas>
        <div id="chatbox"></div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r152/three.min.js"></script>
        <script>
          const canvas = document.getElementById('c');
          const scene = new THREE.Scene();
          const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 1000);
          const renderer = new THREE.WebGLRenderer({{canvas: canvas, alpha: true}});
          renderer.setSize(window.innerWidth, window.innerHeight);
          camera.position.z = 3.5;
          // Head (sphere) + simple eyes
          const headGeo = new THREE.SphereGeometry(1.0, 64, 64);
          const headMat = new THREE.MeshStandardMaterial({{color: 0xf5c6a5, roughness: 0.6}});
          const head = new THREE.Mesh(headGeo, headMat);
          scene.add(head);
          const eyeMat = new THREE.MeshStandardMaterial({{color: 0x000000}});
          function makeEye(x) {{
            const e = new THREE.Mesh(new THREE.SphereGeometry(0.09, 16, 16), eyeMat);
            e.position.set(x, 0.12, 0.93);
            return e;
          }}
          scene.add(makeEye(-0.28));
          scene.add(makeEye(0.28));
          // lighting
          const light = new THREE.DirectionalLight(0xffffff, 1.0);
          light.position.set(5,5,5);
          scene.add(light);
          const amb = new THREE.AmbientLight(0xffffff, 0.4);
          scene.add(amb);
          // animate
          let t = 0;
          function animate() {{
            requestAnimationFrame(animate);
            t += 0.01;
            head.rotation.y = Math.sin(t/2) * 0.25;
            head.rotation.x = Math.sin(t/3) * 0.08;
            renderer.render(scene, camera);
          }}
          animate();
          // Resize handler
          window.addEventListener('resize', ()=>{{
            renderer.setSize(window.innerWidth, window.innerHeight);
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
          }});
        </script>
      </body>
    </html>
    """
    components.html(html, height=400, scrolling=False)

# ----------------------------
# Main app layout: left for controls, right for results
# ----------------------------
col1, col2 = st.columns([1, 2])
with col1:
    st.header("Controls")
    disease_selection = st.selectbox("Select Disease Module", DISEASES, index=0)
    show_confidence = st.checkbox("Show Confidence", value=True)
    enable_iot = ENABLE_IOT_SIM

    # Input source selection
    input_mode = st.radio("Input Source", ["Manual", "IoT (sim)"], index=0)
    if input_mode == "IoT (sim)":
        enable_iot = True
    else:
        enable_iot = False

    # If use IoT sim, show simulated data
    if enable_iot:
        iot_payload = fetch_iot_sim(disease_selection)
        st.write("Simulated IoT payload:", iot_payload)

    # Poppler path override
    poppler_path = POPPLER_PATH.strip() or None

with col2:
    st.header(f"🩺 {disease_selection} Diagnosis & Analysis")

# Build inputs
if enable_iot:
    inputs = list(iot_payload.values())
    columns = list(iot_payload.keys())
else:
    inputs, columns = build_inputs_ui(disease_selection)

# Prediction & report
if st.button(f"🔍 Predict {disease_selection}"):
    bundle = model_bundles.get(disease_selection, {"model": DummyModel(), "type": "sklearn", "imputer": None, "scaler": None})
    try:
        df_in = pd.DataFrame([inputs], columns=columns)
        arr = preprocess(bundle, df_in)
        pred, conf = predict(bundle, arr)
        message = "Likely condition detected" if pred else "No condition detected"
        if pred:
            st.error(message)
        else:
            st.success(message)
        if conf is not None and show_confidence:
            st.write(f"Confidence: {np.max(conf):.3f}")

        # Build record
        record = {
            "timestamp": now_iso(),
            "disease": disease_selection,
            "input_columns": columns,
            "input_values": inputs,
            "prediction": int(pred),
            "confidence": float(np.max(conf)) if conf is not None else None,
            "is_dummy_model": bundle.get("is_dummy", True)
        }

        # Persist
        saved_online = False
        if IS_ONLINE and ENABLE_FIREBASE and FIREBASE_SDK_PRESENT:
            saved_online = save_prediction_to_firestore(record)
        if not saved_online:
            saved_local = save_prediction_offline(record)
            if saved_local:
                st.info("Saved prediction locally (offline mode).")

        # AI Doctor analysis
        ai_summary = {}
        if ENABLE_NLP:
            combined = " ".join([f"{c}:{v}" for c, v in zip(columns, inputs)])
            ai_summary = ai_doctor_analysis(combined, prediction_summary=f"{disease_selection} pred={pred}")
            st.subheader("AI Doctor")
            st.write("**Summary:**", ai_summary.get("summary", "N/A"))
            st.write("**Recommendations:**", ai_summary.get("recommendations", "N/A"))
            st.write("**Severity:**", ai_summary.get("severity", "N/A"))
            st.write("**Explain:**", ai_summary.get("explain", "N/A"))

        # PDF report
        df_report = df_in.copy()
        metadata = {"Disease": disease_selection, "Prediction": int(pred), "Confidence": float(np.max(conf)) if conf is not None else "N/A", "Analyzed At": now_iso()}
        if ai_summary:
            metadata["AI_Summary"] = ai_summary.get("summary","")[:250]
            metadata["AI_Severity"] = ai_summary.get("severity","")
        pdf_bytes = generate_pdf(f"{disease_selection} Diagnosis Report", metadata, df_report)
        st.download_button("📥 Download Diagnosis PDF", data=pdf_bytes, file_name=f"{disease_selection}_diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        safe_print(traceback.format_exc())

# ----------------------------
# OCR section
# ----------------------------
st.markdown("---")
st.header("📄 Medical Report Detection (OCR + AI Doctor)")
uploaded = st.file_uploader("Upload Report (PDF/IMAGE)", type=["pdf", "png", "jpg", "jpeg"])

if uploaded is not None:
    try:
        ext = uploaded.name.split(".")[-1].lower()
        tmp_path = TMP_DIR / f"uploaded_{int(time.time())}.{ext}"
        tmp_path.write_bytes(uploaded.read())

        if ext in ("png", "jpg", "jpeg") and OCR_IMG_AVAILABLE and ENABLE_OCR:
            from PIL import Image
            text = pytesseract.image_to_string(Image.open(tmp_path))
        elif ext == "pdf":
            text = extract_text_from_pdf(tmp_path, poppler_path=poppler_path)
        else:
            text = "OCR not available for this format."

        st.text_area("📝 Extracted Report Text", value=text[:200000], height=300)

        rec = {"timestamp": now_iso(), "source_file": uploaded.name, "extracted_text": text[:10000]}
        saved = False
        if IS_ONLINE and ENABLE_FIREBASE and FIREBASE_SDK_PRESENT and firebase_firestore_client is not None:
            try:
                firebase_firestore_client.collection("ocr_reports").add(rec)
                saved = True
            except Exception:
                saved = False
        if not saved:
            save_ocr_offline(rec)
            st.info("Saved OCR report locally (offline).")

        if ENABLE_NLP:
            ai = ai_doctor_analysis(text, prediction_summary=None)
            st.subheader("AI Doctor Analysis")
            st.write("**Summary:**", ai.get("summary", "N/A"))
            st.write("**Recommendations:**", ai.get("recommendations", "N/A"))
            st.write("**Severity:**", ai.get("severity", "N/A"))
            st.write("**Explanation:**", ai.get("explain", "N/A"))

        # provide PDF of extracted text
        try:
            buffer = io.BytesIO()
            if REPORTLAB_AVAILABLE:
                doc = SimpleDocTemplate(buffer)
                styles = getSampleStyleSheet()
                story = []
                max_chars = 2500
                for i in range(0, len(text), max_chars):
                    story.append(Paragraph(text[i:i+max_chars], styles["Normal"]))
                    story.append(PageBreak())
                doc.build(story)
                pdf_bytes = buffer.getvalue()
            else:
                pdf_bytes = ("Extracted Text:\n\n" + text[:20000]).encode("utf-8")
            st.download_button("📥 Download OCR Report (PDF)", data=pdf_bytes, file_name=f"OCR_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
        except Exception as e:
            st.warning(f"PDF generation failed: {e}")

    except Exception as e:
        st.error(f"OCR processing failed: {e}")
        safe_print(traceback.format_exc())
else:
    st.info("Upload a medical report (PDF/image) to extract and analyze.")

# ----------------------------
# 3D Chatbot panel (chat interacts with AI Doctor)
# ----------------------------
st.markdown("---")
st.header("💬 Virtual 3D Chatbot (AI Doctor assistant)")

if SHOW_3D_CHATBOT:
    # Left: 3D avatar, Right: chat UI
    c1, c2 = st.columns([1, 2])
    with c1:
        render_3d_chatbot()
    with c2:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        # show chat
        for m in st.session_state.chat_history[-50:]:
            if m["role"] == "user":
                st.markdown(f"<div style='text-align:right; color:white; background:#26A2FF; padding:6px; border-radius:8px; margin:4px;'>{m['text']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align:left; background:#F3F4F6; padding:6px; border-radius:8px; margin:4px;'>{m['text']}</div>", unsafe_allow_html=True)

        user_input = st.text_input("Ask the AI Doctor anything (medical summary, what does report mean, explain):", key="chat_input")
        if st.button("Send", key="send_btn"):
            if user_input and user_input.strip():
                st.session_state.chat_history.append({"role": "user", "text": user_input})
                # call AI Doctor
                try:
                    response = ai_doctor_analysis(user_input, prediction_summary=None)
                    out = f"Summary: {response.get('summary','N/A')}\n\nRecommendations: {response.get('recommendations','N/A')}\nSeverity: {response.get('severity','N/A')}\nExplain: {response.get('explain','')}"
                except Exception as e:
                    out = f"AI Doctor error: {e}"
                st.session_state.chat_history.append({"role": "assistant", "text": out})
                # clear input
                st.session_state.chat_input = ""
        st.caption("Note: This chatbot is for educational/demo use only. Not a replacement for professional medical advice.")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("© 2025 Akshat Sharma | AI Health Assistant v4.1 — Offline/Online ready")