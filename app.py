# app.py
# HealthAI Suite - Intelligent Analytics for Patient Care (with TTS modes)

import streamlit as st
import os, sys, tempfile, uuid, time, re, io, asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from typing import List, Dict, Any
import torch
import torchvision.transforms as T
from PIL import Image
import datetime

# Fix sqlite3 for Chroma
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

# Gemini
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

KNOWLEDGE_BASE_TEXT = """
### HealthAI Suite Modules Overview
... (same KB text as before, unchanged for brevity) ...
"""

# -------------------------
# Cached init
# -------------------------
@st.cache_resource(show_spinner=False)
def initialize_rag_dependencies():
    try:
        db_path = tempfile.mkdtemp()
        db_client = chromadb.PersistentClient(path=db_path)
        model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        if GEMINI_API_KEY and genai:
            gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            google_search_tool = [types.Tool(google_search={})]
        else:
            gemini_client = None
            google_search_tool = None
            if GEMINI_API_KEY is None:
                st.warning("GEMINI_API_KEY missing in secrets.")
        return db_client, model, gemini_client, google_search_tool
    except Exception as e:
        st.error(f"Init error: {e}")
        st.stop()

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
    clf = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=50, random_state=42))])
    reg = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestRegressor(n_estimators=50, random_state=42))])
    return clf, reg

# -------------------------
# RAG/KB helpers
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
    if 'db_client' not in st.session_state:
        st.session_state.db_client, _, _, _ = initialize_rag_dependencies()
    db_client = st.session_state.db_client
    try:
        db_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    with st.spinner("Processing KB..."):
        docs = split_documents(KNOWLEDGE_BASE_TEXT)
        process_and_store_documents(docs)
    kb_count = get_collection().count()
    st.session_state["messages_rag"] = [
        {"role": "assistant", "content": f"KB reset. Loaded {kb_count} chunks. Ask me anything!"}
    ]
    st.session_state.module_interaction_log = {}
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
# TTS utilities
# -------------------------
def tts_gemini(text: str, voice_name="Kore", container="MP3"):
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
            ),
            audio_config=types.AudioConfig(
                container=types.AudioContainer(container=container)
            ),
        )
        resp = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=text,
            config=cfg,
        )
        return resp.binary, None
    except Exception as e:
        return None, str(e)

async def _tts_edge_async(text: str, voice="en-US-AriaNeural", rate="0%", pitch="0st"):
    if not edge_tts:
        return None
    out = io.BytesIO()
    comm = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
    async for chunk in comm.stream():
        if chunk[2]:
            out.write(chunk[2])
    return out.getvalue()

def tts_edge(text: str, voice="en-US-AriaNeural", rate="0%", pitch="0st"):
    try:
        return asyncio.run(_tts_edge_async(text, voice, rate, pitch)), None
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
    # Returns (bytes, mime, error)
    if engine == "Gemini TTS":
        audio, err = tts_gemini(text, voice_name="Kore", container="MP3")
        return (audio, "audio/mp3", err)
    if engine == "Edge-TTS":
        audio, err = tts_edge(text, voice="en-US-AriaNeural")
        return (audio, "audio/mp3", err)
    if engine == "gTTS":
        audio, err = tts_gtts(text, lang=lang_code)
        return (audio, "audio/mp3", err)
    return (None, None, "Unknown engine")

# -------------------------
# RAG pipeline with dynamic context
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
            "### CURRENT USER SESSION HISTORY (CRITICAL CONTEXT)\n"
            "The user recently performed the following interactions:\n"
            + joined_entries + "\n"
            "Use this history if relevant, otherwise ignore it.\n\n"
        )

    use_external_search = len(relevant_docs) < 3 or collection.count() == 0

    if use_external_search:
        system_instruction = (
            f"You are a friendly health consultant. "
            f"Use Google Search tool to ground answers for: '{query}'. "
            f"Provide citations. Respond in {selected_language}."
        )
        fallback_query = (
            f"{dynamic_context}"
            f"Provide a detailed, consultant-style answer to: {query}. "
            "Support facts with search results or the provided history."
        )
        response_json = call_gemini_api(
            prompt=fallback_query,
            system_instruction=system_instruction,
            tools=st.session_state.google_search_tool
        )
    else:
        kb_context = "\n".join(relevant_docs)
        rag_system_instruction = (
            "You are a medical assistant. Use static KB context AND session history. "
            "If not found, say you lack specific KB info. "
            f"Respond in {selected_language}."
        )
        prompt = (
            f"{dynamic_context}"
            f"### STATIC KNOWLEDGE BASE CONTEXT\n{kb_context}\n\n"
            f"Question: {query}\n\nAnswer:"
        )
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
    pred_label = int(np.argmax(probs))
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
# Sidebar: global controls
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

# Response mode + TTS engine (applies to chat modules)
st.sidebar.markdown("---")
st.sidebar.subheader("Response Options")
resp_mode = st.sidebar.selectbox("Response mode", ["Text", "Voice"])
tts_engine = st.sidebar.selectbox("TTS engine", ["Gemini TTS", "Edge-TTS", "gTTS"])
st.sidebar.caption("Voice mode renders audio and a download button.")

# Language (for RAG output language and gTTS language code)
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = "English"
lang_display = st.sidebar.selectbox("Answer Language (RAG & gTTS)", list(LANGUAGE_DICT.keys()), index=0)
st.session_state.selected_language = lang_display
lang_code = LANGUAGE_DICT.get(lang_display, "en")

# Session for dynamic context
if 'module_interaction_log' not in st.session_state:
    st.session_state.module_interaction_log = {}

# Initialize RAG/Gemini client lazily for chat modules
if menu in ("🧠 RAG Chatbot", "💡 Together Chat Assistant"):
    if 'db_client' not in st.session_state:
        st.session_state.db_client, st.session_state.model, st.session_state.gemini_client, st.session_state.google_search_tool = initialize_rag_dependencies()
        if st.session_state.db_client and get_collection().count() == 0:
            with st.spinner("Loading KB..."):
                process_and_store_documents(split_documents(KNOWLEDGE_BASE_TEXT))
            st.toast(f"Loaded {get_collection().count()} KB chunks.", icon="📚")

# Load other models
text_tok, text_model = load_text_classifier()
sent_tok, sent_model = load_sentiment_model()
demo_clf, demo_reg = load_tabular_models()

# -------------------------
# Modules
# -------------------------
if menu == "🧑‍⚕️ Risk Stratification":
    st.title("Risk Stratification 🧑‍⚕️")
    st.write("Predict a patient's risk level based on key health indicators.")
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
        result_str = f"Risk: {label} (Score: {score}). Inputs: Age {pdata['age']}, BMI {pdata['bmi']}, Glucose {pdata['glucose']}."
        st.session_state.module_interaction_log[menu] = {"timestamp": current_time, "result": result_str}

elif menu == "⏱ Length of Stay Prediction":
    st.title("Length of Stay Prediction ⏱")
    st.write("Predicts expected hospital length of stay (days).")
    submitted, pdata = patient_input_form("los")
    if submitted:
        los_est = 3.0 + (pdata['age']/30.0) + (pdata['bmi']/40.0) + (pdata['glucose']/200.0)
        los_est_rounded = int(round(los_est))
        st.success(f"Predicted LOS: {los_est_rounded} days")
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state.module_interaction_log[menu] = {"timestamp": current_time, "result": f"LOS {los_est_rounded} days."}

elif menu == "👥 Patient Segmentation":
    st.title("Patient Segmentation 👥")
    st.write("Assigns a patient to a cohort and visualizes clustering in 3D.")
    submitted, pdata = patient_input_form("seg")
    if submitted:
        X_new = preprocess_structured_input(pdata)
        rng = np.random.RandomState(42)
        synthetic_data = rng.normal(loc=[50,25,120,80,100,180], scale=[15,5,20,10,30,40], size=(200,6))
        X_all = np.vstack([synthetic_data, X_new])

        scaler = StandardScaler(); Xs = scaler.fit_transform(X_all)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Xs[:-1])
        pred_label_index = kmeans.predict(Xs[-1].reshape(1, -1))[0]
        all_labels = kmeans.predict(Xs)
        cohort_label = f"Cohort {pred_label_index + 1}"

        st.success(f"Assigned Cohort: {cohort_label}")
        st.info("Plot shows PCA-reduced 3D scatter; your patient is highlighted in red.")

        pca = PCA(n_components=3, random_state=42)
        X_pca = pca.fit_transform(Xs)
        df_plot = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2', 'PCA3'])
        df_plot['Cohort'] = [f"Cohort {l+1}" for l in all_labels]
        df_plot['Type'] = ['Existing' for _ in range(len(synthetic_data))] + ['New Patient']

        fig = px.scatter_3d(
            df_plot, x='PCA1', y='PCA2', z='PCA3',
            color='Type', symbol='Type', opacity=0.8,
            title="Patient Segmentation (3D PCA)",
            color_discrete_map={'Existing': 'blue', 'New Patient': 'red'},
            custom_data=['Type','Cohort']
        )
        fig.update_traces(
            marker=dict(size=[10 if t == 'New Patient' else 5 for t in df_plot['Type']]),
            hovertemplate="<b>Type:</b> %{customdata[0]}<br><b>Cohort:</b> %{customdata[1]}<br>PCA1: %{x}<br>PCA2: %{y}<br>PCA3: %{z}<extra></extra>"
        )
        st.plotly_chart(fig, use_container_width=True)

        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state.module_interaction_log[menu] = {"timestamp": current_time, "result": f"Assigned {cohort_label}."}

elif menu == "🩻 Imaging Diagnostics":
    st.title("Imaging Diagnostics 🩻")
    st.write("Simulates AI analysis of an uploaded image.")
    st.info("Placeholder demo.")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        @st.cache_resource
        def dummy_diagnose_image(image):
            diag = np.random.choice(["No Anomaly Detected", "Pneumonia Detected", "Fracture Identified", "Mass Detected"], p=[0.7,0.15,0.1,0.05])
            confidence = np.random.uniform(0.7, 0.99)
            return {"diagnosis": diag, "confidence": confidence}
        if st.button("Run Diagnosis"):
            with st.spinner("Analyzing..."):
                result = dummy_diagnose_image(uploaded_file)
                st.success(f"Diagnosis: {result['diagnosis']} (Confidence: {result['confidence']:.2f})")
                current_time = datetime.datetime.now().strftime("%H:%M:%S")
                st.session_state.module_interaction_log[menu] = {"timestamp": current_time, "result": f"Diag: {result['diagnosis']}."}

elif menu == "📈 Sequence Forecasting":
    st.title("Sequence Forecasting 📈")
    st.write("Predict the next value of a time series.")
    col1, col2 = st.columns(2)
    with col1:
        num_points = st.slider("Number of data points", 5, 50, 15)
    with col2:
        noise_level = st.slider("Noise level", 0.0, 1.0, 0.1)
    if st.button("Generate & Predict"):
        np.random.seed(42)
        trend = np.linspace(50, 80, num_points)
        noise = np.random.normal(0, noise_level * 10, num_points)
        data = trend + noise
        df_seq = pd.DataFrame({"Time": range(1, num_points + 1), "Metric Value": data})
        st.subheader("Generated Series")
        st.line_chart(df_seq.set_index("Time"))
        last_two = data[-2:]; prediction = last_two[1] + (last_two[1] - last_two[0])
        st.success(f"Predicted next value: {prediction:.2f}")
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state.module_interaction_log[menu] = {"timestamp": current_time, "result": f"Next value {prediction:.2f}."}

elif menu == "📝 Clinical Notes Analysis":
    st.title("Clinical Notes Analysis 📝")
    st.write("Analyze clinical notes for tone.")
    notes = st.text_area("Paste clinical notes", height=200)
    if st.button("Analyze Notes"):
        if not notes.strip():
            st.warning("Please paste notes.")
        else:
            res = text_classify(notes, text_tok, text_model, labels=["Anger","Disgust","Fear","Joy","Neutral","Sadness","Surprise"])
            if res['label'] in ('error','unknown'):
                st.error("Analysis failed.")
                desc = "Analysis failed."
            else:
                desc = f"Tone: {res['label']} (Confidence: {res['score']:.2f})"
                st.success(f"Primary tone: {res['label']} ({res['score']:.2f})")
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            snippet = (notes[:30] + "...") if len(notes) > 30 else notes
            st.session_state.module_interaction_log[menu] = {"timestamp": current_time, "result": f"Notes analysis. {desc} Snippet: '{snippet}'."}

elif menu == "🌐 Translator":
    st.title("Translator 🌐")
    col1, col2 = st.columns(2)
    with col1:
        src_lang = st.selectbox("Source Language", list(LANGUAGE_DICT.keys()), index=0)
    with col2:
        tgt_lang = st.selectbox("Target Language", list(LANGUAGE_DICT.keys()), index=1)
    text_to_trans = st.text_area("Text to translate", "Please describe your symptoms and any medications you are taking.")
    if st.button("Translate"):
        src_code = LANGUAGE_DICT.get(src_lang, "en")
        tgt_code = LANGUAGE_DICT.get(tgt_lang, "en")
        with st.spinner("Translating..."):
            translated_text = translate_text(text_to_trans, src_code, tgt_code)
            st.success("Translated Text:")
            st.write(translated_text)
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            snippet = (translated_text[:30] + "...") if len(translated_text) > 30 else translated_text
            st.session_state.module_interaction_log[menu] = {"timestamp": current_time, "result": f"Translated {src_lang}->{tgt_lang}: '{snippet}'."}

elif menu == "💬 Sentiment Analysis":
    st.title("Patient Feedback Sentiment Analysis 💬")
    patient_feedback = st.text_area("Patient Feedback", "The nurse was very helpful, but the wait time was too long.")
    if st.button("Analyze Sentiment"):
        if not patient_feedback.strip():
            st.warning("Please provide feedback.")
        else:
            sentiment_result = sentiment_text(patient_feedback, sent_tok, sent_model)
            if sentiment_result['label'] == 'unknown':
                st.error("Model not loaded.")
                sentiment_label, confidence = "Failed", 0.0
            else:
                sentiment_label, confidence = sentiment_result['label'], sentiment_result['score']
                st.success(f"Sentiment: {sentiment_label} ({confidence:.2f})")
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            snippet = (patient_feedback[:30] + "...") if len(patient_feedback) > 30 else patient_feedback
            st.session_state.module_interaction_log[menu] = {"timestamp": current_time, "result": f"Sentiment {sentiment_label} ({confidence:.2f}). '{snippet}'."}

elif menu == "💡 Together Chat Assistant":
    st.title("Together Chat Assistant 💡 (Powered by Gemini)")
    if not GEMINI_API_KEY:
        st.error("Set GEMINI_API_KEY in secrets.toml to use this module.")
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
                response_json = call_gemini_api(
                    prompt=prompt,
                    model_name="gemini-2.5-flash",
                    system_instruction="You are a helpful and medically accurate general health assistant. Keep answers concise."
                )
                answer_text = response_json.get('response', response_json.get('error', "An unknown error occurred."))
                st.write(answer_text)
                # Voice mode handling
                if resp_mode == "Voice":
                    audio_bytes, mime, err = synthesize(answer_text, tts_engine, lang_code)
                    if audio_bytes:
                        st.audio(io.BytesIO(audio_bytes), format=mime)
                        st.download_button("Download speech", data=audio_bytes, file_name="answer.mp3", mime=mime)
                    else:
                        st.warning(f"TTS failed: {err}")
                st.session_state.messages_assistant.append({"role": "assistant", "content": answer_text})

elif menu == "🧠 RAG Chatbot":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Knowledge Base")
    kb_count = get_collection().count()
    st.sidebar.info(f"KB Chunks: {kb_count}")
    if st.sidebar.button("Reset/Reload Default KB"):
        clear_and_reload_kb()
        st.toast("KB reset and reloaded!", icon="🔄")

    st.markdown("## Health RAG Chatbot 🧠 (Context-Aware)")
    st.markdown("Specialized Health Consultant AI using a local KB with Google Search fallback and session-aware context.")

    if "messages_rag" not in st.session_state:
        st.session_state["messages_rag"] = [
            {"role": "assistant", "content": "Hello! I'm your context-aware RAG medical assistant. Ask any health question."}
        ]
    for msg in st.session_state.messages_rag:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask a health question..."):
        st.session_state.messages_rag.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating..."):
                answer_text = rag_pipeline(prompt, st.session_state.selected_language)
                st.write(answer_text)
                # Voice mode handling
                if resp_mode == "Voice":
                    audio_bytes, mime, err = synthesize(answer_text, tts_engine, lang_code)
                    if audio_bytes:
                        st.audio(io.BytesIO(audio_bytes), format=mime)
                        st.download_button("Download speech", data=audio_bytes, file_name="answer.mp3", mime=mime)
                    else:
                        st.warning(f"TTS failed: {err}")
                st.session_state.messages_rag.append({"role": "assistant", "content": answer_text})
