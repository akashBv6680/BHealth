# app.py
# HealthAI Suite - Intelligent Analytics for Patient Care (TTS fixes)

import streamlit as st
import os, sys, tempfile, uuid, time, re, io, asyncio, datetime
import numpy as np
import pandas as pd
import plotly.express as px
from typing import Dict, Any
import torch
from PIL import Image

# sqlite fix for Chroma
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules['pysqlite3']
except ImportError:
    pass

# RAG deps
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# HF models
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    MarianMTModel,
    MarianTokenizer,
)

# ML
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Gemini SDK (python-genai)
try:
    from google import genai
    from google.genai.errors import APIError
    from google.genai import types
except ImportError:
    genai = None
    APIError = None
    types = None

# Optional TTS engines
try:
    import edge_tts
except Exception:
    edge_tts = None
try:
    from gtts import gTTS
except Exception:
    gTTS = None

# -------------------------
# App config
# -------------------------
st.set_page_config(page_title="HealthAI Suite", page_icon="🩺", layout="wide")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

LANGUAGE_DICT = {
    "English": "en", "Spanish": "es", "Arabic": "ar", "French": "fr", "German": "de", "Hindi": "hi",
    "Tamil": "ta", "Bengali": "bn", "Japanese": "ja", "Korean": "ko", "Russian": "ru",
    "Chinese (Simplified)": "zh-Hans", "Portuguese": "pt", "Italian": "it", "Dutch": "nl", "Turkish": "tr"
}

COLLECTION_NAME = "rag_documents"



# --- MASSIVELY EXPANDED KNOWLEDGE BASE (MULTILINGUAL - NO CHANGE) ---
KNOWLEDGE_BASE_TEXT = """
### HealthAI Suite Modules Overview
The HealthAI Suite is an integrated platform offering ten key modules for intelligent healthcare analytics. These modules cover risk assessment, predictive modeling, patient management, clinical text processing, and diagnostic support. Clients and users can navigate to any module via the sidebar to perform a specific function, or use the RAG Chatbot for a general consultation.

### 🧑‍⚕️ Risk Stratification Module
This module assesses a patient's **overall health risk** (Low, Moderate, or High) based on structured inputs like age, BMI, blood pressure (SBP/DBP), glucose, cholesterol, and smoking status. It helps clinicians quickly identify patients who may require intensive intervention or monitoring.

### ⏱ Length of Stay Prediction Module
This tool estimates the **expected number of days** a patient will remain hospitalized. It uses factors like admission metrics (age, vital signs, initial labs) to forecast resource utilization and aid in discharge planning and bed management.

### 👥 Patient Segmentation Module
This module uses clustering techniques (like K-Means) on various patient health metrics to **group patients into distinct cohorts**. This helps identify common patient profiles, tailor care pathways, and analyze outcomes based on specific, shared characteristics.

### 🩻 Imaging Diagnostics Module
This module is designed to support clinicians by performing **AI-driven analysis of medical images** (e.g., X-rays, CT scans). It helps detect, classify, and localize pathologies (like pneumonia, fractures, or masses), offering a diagnostic prediction and a confidence score.

### 📈 Sequence Forecasting Module
This module is specialized for **time-series data**. It analyzes a historical sequence of a patient's metric (e.g., blood sugar, temperature) and predicts the **next value in the sequence**. This is useful for anticipating future patient states.

### 📝 Clinical Notes Analysis Module
This tool processes **unstructured clinical text** (doctor's notes, discharge summaries) to extract meaningful insights. It can classify the type of note or analyze the emotional tone to aid in quality control and information extraction.

### 🌐 Translator Module
The Translator provides real-time, bi-directional translation for **clinical and patient-facing text** between various languages (e.g., English, Spanish, Hindi). This is crucial for improving communication between multilingual staff and patients.

### 💬 Sentiment Analysis Module
This module is dedicated to analyzing **patient feedback** or reviews to automatically determine the **emotional tone** (Positive, Negative, or Neutral). It helps healthcare providers quickly gauge patient satisfaction and pinpoint areas for operational improvement.

### 💡 Together Chat Assistant Module
This is a **general-purpose chat assistant** powered by the Google Gemini model. It is designed for quick, general inquiries and conversation about non-specific health topics, acting as a general-knowledge resource.

### 🧠 RAG Chatbot Module
This is a **specialized Health Consultant AI**. It uses **Retrieval-Augmented Generation (RAG)**, meaning it first consults its fixed internal **Knowledge Base (KB)** (containing detailed common health topics) and, if the answer is outside the KB, it automatically uses the **Google Search tool** to find current information and provides **citations** for reliability. It acts as the primary expert query engine.

### Common Health Topics (Detailed KB)
The RAG chatbot's internal KB also contains detailed information on common conditions.
**Common Cold and Flu:** Viral infections of the respiratory tract. Cold is milder; flu is severe, often with fever and body aches. Management: rest, hydration, OTC meds.
**Diabetes (Type 2) Management:** Chronic high blood sugar. Managed by diet, exercise (150 mins/week), Metformin, and monitoring. Risks: heart disease, nerve damage.
**Hypertension (High Blood Pressure):** High force against artery walls. Often silent. Treatment: DASH diet, exercise, ACE inhibitors, diuretics. Normal BP < 120/80 mmHg.
**Asthma Control:** Narrowing and swelling of airways. Symptoms: wheezing, shortness of breath. Treatment: Reliever (quick) and Preventer (daily) inhalers.
**Mental Health:** Depression (persistent sadness) and Anxiety (excessive worry). Treated with psychotherapy (CBT) and medication (antidepressants).
"""

# -------------------------
# Cached init
# -------------------------
@st.cache_resource(show_spinner=False)
def initialize_rag_dependencies():
    db_path = tempfile.mkdtemp()
    db_client = chromadb.PersistentClient(path=db_path)
    model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
    if GEMINI_API_KEY and genai:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        google_search_tool = [types.Tool(google_search={})]
    else:
        gemini_client = None
        google_search_tool = None
    return db_client, model, gemini_client, google_search_tool

@st.cache_resource(show_spinner=False)
def load_text_classifier(model_name="bhadresh-savani/bert-base-uncased-emotion"):
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        m = AutoModelForSequenceClassification.from_pretrained(model_name)
        m.eval()
        return tok, m
    except Exception:
        return None, None

@st.cache_resource(show_spinner=False)
def load_translation_model(src_lang="en", tgt_lang="hi"):
    pair = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    try:
        tkn = MarianTokenizer.from_pretrained(pair)
        m = MarianMTModel.from_pretrained(pair)
        m.eval()
        return tkn, m
    except Exception:
        return None, None

@st.cache_resource(show_spinner=False)
def load_sentiment_model(model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        m = AutoModelForSequenceClassification.from_pretrained(model_name)
        m.eval()
        return tok, m
    except Exception:
        return None, None

@st.cache_resource(show_spinner=False)
def load_tabular_models():
    from sklearn.metrics import make_scorer
    clf = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=50, random_state=42))])
    reg = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestRegressor(n_estimators=50, random_state=42))])
    return clf, reg

# -------------------------
# RAG helpers
# -------------------------
def get_collection():
    if 'db_client' not in st.session_state:
        st.session_state.db_client, st.session_state.model, st.session_state.gemini_client, st.session_state.google_search_tool = initialize_rag_dependencies()
    return st.session_state.db_client.get_or_create_collection(name=COLLECTION_NAME)

def split_documents(text_data, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, is_separator_regex=False)
    return splitter.split_text(text_data)

def process_and_store_documents(documents):
    collection = get_collection()
    model = st.session_state.model
    embeddings = model.encode(documents).tolist()
    ids = [str(uuid.uuid4()) for _ in documents]
    collection.add(documents=documents, embeddings=embeddings, ids=ids)

def retrieve_documents(query, n_results=5):
    collection = get_collection()
    model = st.session_state.model
    q_emb = model.encode(query).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=n_results, include=['documents', 'distances'])
    return results['documents'][0] if results['documents'] else []

def clear_and_reload_kb():
    db_client = st.session_state.db_client if 'db_client' in st.session_state else initialize_rag_dependencies()[0]
    try:
        db_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    with st.spinner("Processing KB..."):
        docs = split_documents(KNOWLEDGE_BASE_TEXT)
        process_and_store_documents(docs)
    st.rerun()

def call_gemini_api(prompt, model_name="gemini-2.5-flash", system_instruction="You are a helpful health assistant.", tools=None, max_retries=5):
    if not st.session_state.get('gemini_client'):
        return {"error": "Gemini client not configured."}
    retry_delay = 1
    for i in range(max_retries):
        try:
            cfg = types.GenerateContentConfig(system_instruction=system_instruction, tools=tools)
            resp = st.session_state.gemini_client.models.generate_content(model=model_name, contents=prompt, config=cfg)
            return {"response": resp.text}
        except APIError as e:
            if i < max_retries - 1:
                time.sleep(retry_delay); retry_delay *= 2; continue
            return {"error": str(e)}
        except Exception as e:
            return {"error": str(e)}

# -------------------------
# TTS utilities (updated)
# -------------------------
def tts_gemini(text: str, voice_name="Kore"):
    """
    SDK-compatible Gemini TTS call without types.AudioConfig (older SDKs lack it).
    Returns (bytes, error)
    """
    if not (genai and GEMINI_API_KEY):
        return None, "Gemini TTS not available."
    try:
        client = st.session_state.get("gemini_client") or genai.Client(api_key=GEMINI_API_KEY)
        cfg = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                )
            )
        )
        resp = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=text,
            config=cfg,
        )
        return resp.binary, None
    except Exception as e:
        return None, str(e)

async def _edge_async(text: str, voice="en-US-AriaNeural", rate=None):
    if not edge_tts:
        return None
    # Normalize rate: accept "0%" and convert to "+0%"
    kwargs = {"text": text, "voice": voice}
    if rate:
        r = rate.strip()
        if r == "0%" or r == "0":
            r = "+0%"
        elif not r.startswith(("+","-")):
            r = f"+{r}"
        kwargs["rate"] = r
    comm = edge_tts.Communicate(**kwargs)
    out = io.BytesIO()
    async for chunk in comm.stream():
        if chunk[2]:
            out.write(chunk[2])
    return out.getvalue()

def tts_edge(text: str, voice="en-US-AriaNeural", rate=None):
    try:
        return asyncio.run(_edge_async(text, voice, rate)), None
    except Exception as e:
        return None, str(e)

def tts_gtts(text: str, lang="en"):
    if not gTTS:
        return None, "gTTS not available."
    try:
        buf = io.BytesIO()
        gTTS(text, lang=lang).write_to_fp(buf)
        return buf.getvalue(), None
    except Exception as e:
        return None, str(e)

def synthesize(text: str, engine: str, lang_code="en"):
    """
    Returns (audio_bytes, mime, error)
    Language-aware voices are handled here:
      - gTTS: uses lang_code directly.
      - Edge: picks a matching voice if available; else falls back to gTTS.
      - Gemini: uses a neutral prebuilt voice (language inferred from text).
    """
    edge_voice_map = {
        "en": "en-US-AriaNeural", "hi": "hi-IN-SwaraNeural", "ta": "ta-IN-PallaviNeural", "bn": "bn-IN-BashkarNeural",
        "es": "es-ES-AlvaroNeural", "fr": "fr-FR-DeniseNeural", "de": "de-DE-KatjaNeural", "ar": "ar-SA-HamedNeural",
        "zh-Hans": "zh-CN-XiaoxiaoNeural", "zh-cn": "zh-CN-XiaoxiaoNeural", "ja": "ja-JP-NanamiNeural",
        "ko": "ko-KR-SunHiNeural", "pt": "pt-PT-FernandaNeural", "it": "it-IT-ElsaNeural", "nl": "nl-NL-ColetteNeural",
        "tr": "tr-TR-AhmetNeural", "ru": "ru-RU-DariyaNeural"
    }

    if engine == "Gemini TTS":
        audio, err = tts_gemini(text, voice_name="Kore")
        return (audio, "audio/mp3", err)

    if engine == "Edge-TTS":
        voice = edge_voice_map.get(lang_code, "en-US-AriaNeural")
        audio, err = tts_edge(text, voice=voice, rate="+0%")
        if audio:
            return (audio, "audio/mp3", None)
        audio2, err2 = tts_gtts(text, lang=lang_code if lang_code else "en")
        return (audio2, "audio/mp3", err or err2)

    if engine == "gTTS":
        audio, err = tts_gtts(text, lang=lang_code if lang_code else "en")
        return (audio, "audio/mp3", err)

    return (None, None, "Unknown engine")

# -------------------------
# RAG pipeline (unchanged logic, safe join)
# -------------------------
def rag_pipeline(query, selected_language):
    collection = get_collection()
    relevant_docs = retrieve_documents(query)

    dynamic_context = ""
    if st.session_state.get("module_interaction_log"):
        log_entries = []
        for module, data in st.session_state.module_interaction_log.items():
            ts = data.get('timestamp', '')
            res = data.get('result', '')
            log_entries.append(f" - {module} (Last Run: {ts}) - Result: {res}")
        joined_entries = "\n".join(log_entries)
        dynamic_context = (
            "### CURRENT USER SESSION HISTORY\n"
            "Recent interactions:\n" + joined_entries + "\n\n"
        )

    use_external_search = len(relevant_docs) < 3 or collection.count() == 0

    if use_external_search:
        system_instruction = (
            f"You are a friendly health consultant. Use Google Search tool if needed. "
            f"Respond in {selected_language} with clear advice and citations when searching."
        )
        fallback_query = f"{dynamic_context}Answer the user's health question: {query}"
        response_json = call_gemini_api(
            prompt=fallback_query,
            system_instruction=system_instruction,
            tools=st.session_state.google_search_tool
        )
    else:
        kb_context = "\n".join(relevant_docs)
        rag_system_instruction = (
            "You are a medical assistant. Use the KB context and session history. "
            f"Respond in {selected_language}."
        )
        prompt = f"{dynamic_context}### KB\n{kb_context}\n\nQuestion: {query}\n\nAnswer:"
        response_json = call_gemini_api(prompt, system_instruction=rag_system_instruction)

    if 'error' in response_json:
        return f"Generation error: {response_json['error']}"
    return response_json.get('response', "No response text.")

# -------------------------
# Utilities
# -------------------------
def patient_input_form(key_prefix="p"):
    with st.form(key=f"form_{key_prefix}"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=45, key=f"{key_prefix}_age")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], key=f"{key_prefix}_gender")
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, key=f"{key_prefix}_bmi")
            sbp = st.number_input("Systolic BP", min_value=60, max_value=250, value=120, key=f"{key_prefix}_sbp")
        with col2:
            dbp = st.number_input("Diastolic BP", min_value=40, max_value=160, value=80, key=f"{key_prefix}_dbp")
            glucose = st.number_input("Glucose (mg/dL)", min_value=40, max_value=400, value=100, key=f"{key_prefix}_glucose")
            cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=500, value=180, key=f"{key_prefix}_cholesterol")
            smoker = st.selectbox("Smoker", ["No", "Yes"], index=0, key=f"{key_prefix}_smoker")
        submitted = st.form_submit_button("Run Analysis")
    data = {
        "age": int(age), "gender": gender, "bmi": float(bmi), "sbp": float(sbp), "dbp": float(dbp),
        "glucose": float(glucose), "cholesterol": float(cholesterol), "smoker": smoker == "Yes"
    }
    return submitted, data

def text_classify(text: str, tokenizer, model, labels=None):
    if tokenizer is None or model is None:
        return {"label": "unknown", "score": 0.0}
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        lbl = labels[pred] if labels else str(pred)
        return {"label": lbl, "score": float(probs[pred])}
    except Exception:
        return {"label": "error", "score": 0.0}

def translate_text(text: str, src: str, tgt: str):
    tkn, m = load_translation_model(src, tgt)
    if tkn is None or m is None:
        return "Translation model not available; returning original text."
    inputs = tkn([text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        generated = m.generate(**inputs)
    return tkn.batch_decode(generated, skip_special_tokens=True)[0]

def sentiment_text(text: str, tokenizer, model):
    if tokenizer is None or model is None:
        return {"label": "unknown", "score": 0.0}
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    pred_label = np.argmax(probs)
    labels = ["Negative", "Neutral", "Positive"]
    return {"label": labels[pred_label], "score": float(probs[pred_label])}

def preprocess_structured_input(data: Dict[str, Any]):
    numeric_keys = ["age", "bmi", "sbp", "dbp", "glucose", "cholesterol"]
    vals = []
    for k in numeric_keys:
        v = data.get(k, 0.0)
        try:
            vals.append(float(v))
        except Exception:
            vals.append(0.0)
    return np.array(vals).reshape(1, -1)

# -------------------------
# Sidebar and state
# -------------------------
st.sidebar.title("HealthAI Suite")
menu = st.sidebar.radio("Select Module", [
    "🧑‍⚕️ Risk Stratification",
    "⏱ Length of Stay Prediction",
    "👥 Patient Segmentation",
    "🩻 Imaging Diagnostics",
    "📈 Sequence Forecasting",
    "📝 Clinical Notes Analysis",
    "🌐 Translator",
    "💬 Sentiment Analysis",
    "💡 Together Chat Assistant",
    "🧠 RAG Chatbot"
])

st.sidebar.markdown("---")
st.sidebar.subheader("Response Options")
resp_mode = st.sidebar.selectbox("Response mode", ["Text", "Voice"])
tts_engine = st.sidebar.selectbox("TTS engine", ["Gemini TTS", "Edge-TTS", "gTTS"])
st.sidebar.caption("Voice mode plays audio inline.")

if 'selected_language' not in st.session_state:
    st.session_state.selected_language = "English"
lang_display = st.sidebar.selectbox("Answer Language (RAG & gTTS)", list(LANGUAGE_DICT.keys()), index=0)
st.session_state.selected_language = lang_display
lang_code = LANGUAGE_DICT.get(lang_display, "en")

if 'module_interaction_log' not in st.session_state:
    st.session_state.module_interaction_log = {}

if menu in ("🧠 RAG Chatbot", "💡 Together Chat Assistant"):
    if 'db_client' not in st.session_state:
        st.session_state.db_client, st.session_state.model, st.session_state.gemini_client, st.session_state.google_search_tool = initialize_rag_dependencies()
        if get_collection().count() == 0:
            process_and_store_documents(split_documents(KNOWLEDGE_BASE_TEXT))
            st.toast(f"Loaded {get_collection().count()} KB chunks.", icon="📚")

text_tok, text_model = load_text_classifier()
sent_tok, sent_model = load_sentiment_model()
demo_clf, demo_reg = load_tabular_models()

# -------------------------
# Modules
# -------------------------
if menu == "🧑‍⚕️ Risk Stratification":
    st.title("Risk Stratification 🧑‍⚕️")
    submitted, pdata = patient_input_form("risk")
    if submitted:
        score = 0
        score += (pdata['age'] >= 60) * 2 + (45 <= pdata['age'] < 60) * 1
        score += (pdata['bmi'] >= 30) * 2 + (25 <= pdata['bmi'] < 30) * 1
        score += (pdata['sbp'] >= 140) * 2 + (130 <= pdata['sbp'] < 140) * 1
        score += (pdata['glucose'] >= 126) * 2 + (110 <= pdata['glucose'] < 126) * 1
        score += (1 if pdata['smoker'] else 0)
        label = "Low Risk" if score <= 1 else ("Moderate Risk" if score <= 3 else "High Risk")
        st.success(f"Predicted Risk Level: {label} (Score: {score})")
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state.module_interaction_log[menu] = {"timestamp": current_time, "result": f"Risk: {label} (Score {score})."}

elif menu == "⏱ Length of Stay Prediction":
    st.title("Length of Stay Prediction ⏱")
    submitted, pdata = patient_input_form("los")
    if submitted:
        los_est = 3.0 + (pdata['age']/30.0) + (pdata['bmi']/40.0) + (pdata['glucose']/200.0)
        st.success(f"Predicted LOS: {int(round(los_est))} days")
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state.module_interaction_log[menu] = {"timestamp": current_time, "result": f"LOS {int(round(los_est))} days."}

elif menu == "👥 Patient Segmentation":
    st.title("Patient Segmentation 👥")
    submitted, pdata = patient_input_form("seg")
    if submitted:
        X_new = preprocess_structured_input(pdata)
        rng = np.random.RandomState(42)
        synthetic = rng.normal(loc=[50,25,120,80,100,180], scale=[15,5,20,10,30,40], size=(200,6))
        X_all = np.vstack([synthetic, X_new])
        scaler = StandardScaler(); Xs = scaler.fit_transform(X_all)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Xs[:-1])
        label_idx = kmeans.predict(Xs[-1].reshape(1,-1))[0]
        all_labels = kmeans.predict(Xs)
        cohort = f"Cohort {label_idx+1}"
        st.success(f"Assigned Cohort: {cohort}")
        pca = PCA(n_components=3, random_state=42); Xp = pca.fit_transform(Xs)
        df = pd.DataFrame(Xp, columns=['PCA1','PCA2','PCA3'])
        df['Cohort'] = [f"Cohort {l+1}" for l in all_labels]
        df['Type'] = ['Existing']*len(synthetic) + ['New Patient']
        fig = px.scatter_3d(df, x='PCA1', y='PCA2', z='PCA3',
                            color='Type', symbol='Type', opacity=0.85,
                            color_discrete_map={'Existing':'blue','New Patient':'red'},
                            custom_data=['Type','Cohort'])
        fig.update_traces(marker=dict(size=[10 if t=='New Patient' else 5 for t in df['Type']]),
                          hovertemplate="<b>Type:</b> %{customdata[0]}<br><b>Cohort:</b> %{customdata[1]}<br>PCA1: %{x}<br>PCA2: %{y}<br>PCA3: %{z}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state.module_interaction_log[menu] = {"timestamp": current_time, "result": f"Assigned {cohort}."}

elif menu == "🩻 Imaging Diagnostics":
    st.title("Imaging Diagnostics 🩻")
    st.info("Placeholder demo.")
    up = st.file_uploader("Upload image", type=["png","jpg","jpeg"])
    if up:
        st.image(up, use_column_width=True)
        @st.cache_resource
        def dummy(img): 
            import numpy as np
            return {"diagnosis": np.random.choice(["No Anomaly","Pneumonia","Fracture","Mass"]),
                    "confidence": float(np.random.uniform(0.7,0.99))}
        if st.button("Run Diagnosis"):
            r = dummy(up)
            st.success(f"Diagnosis: {r['diagnosis']} (Confidence: {r['confidence']:.2f})")
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            st.session_state.module_interaction_log[menu] = {"timestamp": current_time, "result": f"Diag {r['diagnosis']}."}

elif menu == "📈 Sequence Forecasting":
    st.title("Sequence Forecasting 📈")
    c1,c2 = st.columns(2)
    with c1: n = st.slider("Data points", 5, 50, 15)
    with c2: noise = st.slider("Noise", 0.0, 1.0, 0.1)
    if st.button("Generate & Predict"):
        np.random.seed(42)
        trend = np.linspace(50,80,n); eps = np.random.normal(0, noise*10, n)
        y = trend + eps
        df = pd.DataFrame({"Time":range(1,n+1),"Value":y})
        st.line_chart(df.set_index("Time"))
        pred = y[-1] + (y[-1]-y[-2])
        st.success(f"Next value: {pred:.2f}")
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state.module_interaction_log[menu] = {"timestamp": current_time, "result": f"Next {pred:.2f}."}

elif menu == "📝 Clinical Notes Analysis":
    st.title("Clinical Notes Analysis 📝")
    notes = st.text_area("Paste clinical notes", height=200)
    if st.button("Analyze Notes"):
        if not notes.strip():
            st.warning("Please paste notes.")
        else:
            res = text_classify(notes, *load_text_classifier(), labels=["Anger","Disgust","Fear","Joy","Neutral","Sadness","Surprise"])
            if res['label'] in ('error','unknown'):
                st.error("Model not loaded.")
                desc = "Analysis failed."
            else:
                desc = f"Tone: {res['label']} ({res['score']:.2f})"
                st.success(f"Primary tone: {res['label']} ({res['score']:.2f})")
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            st.session_state.module_interaction_log[menu] = {"timestamp": current_time, "result": f"Notes: {desc}"}

elif menu == "🌐 Translator":
    st.title("Translator 🌐")
    c1,c2 = st.columns(2)
    with c1: src = st.selectbox("Source", list(LANGUAGE_DICT.keys()), index=0)
    with c2: tgt = st.selectbox("Target", list(LANGUAGE_DICT.keys()), index=1)
    text_in = st.text_area("Text to translate", "Please describe your symptoms and any medications you are taking.")
    if st.button("Translate"):
        tr = translate_text(text_in, LANGUAGE_DICT[src], LANGUAGE_DICT[tgt])
        st.success("Translated:")
        st.write(tr)
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state.module_interaction_log[menu] = {"timestamp": current_time, "result": f"Translated {src}->{tgt}."}

elif menu == "💬 Sentiment Analysis":
    st.title("Patient Feedback Sentiment Analysis 💬")
    fb = st.text_area("Patient Feedback", "The nurse was very helpful, but the wait time was too long.")
    if st.button("Analyze Sentiment"):
        if not fb.strip():
            st.warning("Please provide feedback.")
        else:
            tok, mdl = load_sentiment_model()
            s = sentiment_text(fb, tok, mdl)
            if s['label'] == 'unknown':
                st.error("Model not loaded.")
            else:
                st.success(f"Sentiment: {s['label']} ({s['score']:.2f})")
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            st.session_state.module_interaction_log[menu] = {"timestamp": current_time, "result": f"Sentiment {s['label']}."}

elif menu == "💡 Together Chat Assistant":
    st.title("Together Chat Assistant 💡 (Gemini)")
    if not GEMINI_API_KEY:
        st.error("Set GEMINI_API_KEY in secrets.toml.")
        st.stop()
    if "messages_assistant" not in st.session_state:
        st.session_state["messages_assistant"] = [
            {"role": "assistant", "content": "Hello! I am a general health assistant powered by Google Gemini. How can I help you today?"}
        ]
    for msg in st.session_state.messages_assistant:
        st.chat_message(msg["role"]).write(msg["content"])
    if prompt := st.chat_input("Ask me anything about general health..."):
        st.session_state.messages_assistant.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                r = call_gemini_api(prompt, model_name="gemini-2.5-flash",
                                    system_instruction="You are a concise, medically accurate assistant.")
                answer = r.get('response', r.get('error', "Unknown error."))
                st.write(answer)
                if resp_mode == "Voice":
                    audio, mime, err = synthesize(answer, tts_engine, lang_code)
                    if audio:
                        st.audio(io.BytesIO(audio), format=mime)  # inline playback only
                    else:
                        st.warning(f"TTS failed: {err}")
                st.session_state.messages_assistant.append({"role": "assistant", "content": answer})

elif menu == "🧠 RAG Chatbot":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Knowledge Base")
    kb_count = get_collection().count()
    st.sidebar.info(f"KB Chunks: {kb_count}")
    if st.sidebar.button("Reset/Reload Default KB"):
        clear_and_reload_kb()
    st.markdown("## Health RAG Chatbot 🧠 (Context-Aware)")
    if "messages_rag" not in st.session_state:
        st.session_state["messages_rag"] = [{"role": "assistant", "content": "Hello! I'm your context-aware RAG medical assistant. Ask any health question."}]
    for msg in st.session_state.messages_rag:
        st.chat_message(msg["role"]).write(msg["content"])
    if prompt := st.chat_input("Ask a health question..."):
        st.session_state.messages_rag.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating..."):
                answer = rag_pipeline(prompt, st.session_state.selected_language)
                st.write(answer)
                if resp_mode == "Voice":
                    audio, mime, err = synthesize(answer, tts_engine, lang_code)
                    if audio:
                        st.audio(io.BytesIO(audio), format=mime)  # inline playback only
                    else:
                        st.warning(f"TTS failed: {err}")
                st.session_state.messages_rag.append({"role": "assistant", "content": answer})
