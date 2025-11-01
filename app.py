# app.py — HealthAI Suite with multilingual TTS routing (inline playback only)

import streamlit as st
import os, sys, tempfile, uuid, time, io, asyncio, datetime
import numpy as np
import pandas as pd
import plotly.express as px
from typing import Dict, Any, Optional, Tuple

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

# Gemini SDK
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
    import edge_tts  # Microsoft Edge neural voices
except Exception:
    edge_tts = None
try:
    from gtts import gTTS  # Google Translate TTS
except Exception:
    gTTS = None

# --------------------------------
# App config
# --------------------------------
st.set_page_config(page_title="HealthAI Suite", page_icon="🩺", layout="wide")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

LANGUAGE_DICT = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Bengali": "bn",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Arabic": "ar",
    "Chinese (Simplified)": "zh-cn",
    "Japanese": "ja",
    "Korean": "ko",
    "Portuguese": "pt",
    "Italian": "it",
    "Dutch": "nl",
    "Turkish": "tr",
    "Russian": "ru",
}

# Edge‑TTS voice map (sampled popular voices; extend as needed)
EDGE_VOICE_MAP = {
    "en": "en-US-AriaNeural",         # English (US) [web:74]
    "hi": "hi-IN-SwaraNeural",        # Hindi (India) [web:76][web:77]
    "ta": "ta-IN-PallaviNeural",      # Tamil (India) [web:77]
    "bn": "bn-IN-BashkarNeural",      # Bengali (India) [web:76]
    "es": "es-ES-AlvaroNeural",       # Spanish (Spain) [web:74]
    "fr": "fr-FR-DeniseNeural",       # French (France) [web:74]
    "de": "de-DE-KatjaNeural",        # German (Germany) [web:74]
    "ar": "ar-SA-HamedNeural",        # Arabic (Saudi) [web:76]
    "zh-cn": "zh-CN-XiaoxiaoNeural",  # Chinese (Mainland) [web:74]
    "ja": "ja-JP-NanamiNeural",       # Japanese [web:74]
    "ko": "ko-KR-SunHiNeural",        # Korean [web:74]
    "pt": "pt-PT-FernandaNeural",     # Portuguese (Portugal) [web:74]
    "it": "it-IT-ElsaNeural",         # Italian [web:74]
    "nl": "nl-NL-ColetteNeural",      # Dutch [web:74]
    "tr": "tr-TR-AhmetNeural",        # Turkish [web:74]
    "ru": "ru-RU-DariyaNeural",       # Russian [web:74]
}

# --------------------------------
# Knowledge Base (truncated here)
# --------------------------------
COLLECTION_NAME = "rag_documents"
KB_TEXT = """
### HealthAI Suite Modules Overview
... (same KB content as earlier) ...
"""

# --------------------------------
# Cached init
# --------------------------------
@st.cache_resource(show_spinner=False)
def initialize_rag_dependencies():
    db_path = tempfile.mkdtemp()
    db_client = chromadb.PersistentClient(path=db_path)
    model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
    if GEMINI_API_KEY and genai:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        google_search_tool = [types.Tool(google_search={})]
    else:
        gemini_client, google_search_tool = None, None
    return db_client, model, gemini_client, google_search_tool

# Light caches
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

# --------------------------------
# RAG helpers
# --------------------------------
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

def call_gemini_api(prompt, model_name="gemini-2.5-flash", system_instruction="You are a helpful health assistant.", tools=None):
    if not st.session_state.get('gemini_client'):
        return {"error": "Gemini client not configured."}
    try:
        cfg = types.GenerateContentConfig(system_instruction=system_instruction, tools=tools)
        resp = st.session_state.gemini_client.models.generate_content(model=model_name, contents=prompt, config=cfg)
        return {"response": resp.text}
    except Exception as e:
        return {"error": str(e)}

# --------------------------------
# TTS: language-aware routing
# --------------------------------
def _normalize_edge_rate(rate: Optional[str]) -> Optional[str]:
    if not rate:
        return None
    r = rate.strip()
    if r in ("0", "0%"):
        return "+0%"
    if not (r.startswith("+") or r.startswith("-")):
        if r.endswith("%"):
            return f"+{r}"
        return f"+{r}%"
    return r

async def _edge_async(text: str, voice: str, rate: Optional[str]) -> Optional[bytes]:
    if not edge_tts:
        return None
    kwargs = {"text": text, "voice": voice}
    r_norm = _normalize_edge_rate(rate)
    if r_norm:
        kwargs["rate"] = r_norm
    comm = edge_tts.Communicate(**kwargs)
    out = io.BytesIO()
    async for chunk in comm.stream():
        if chunk[2]:
            out.write(chunk[2])
    return out.getvalue()

def tts_edge(text: str, lang_code: str, rate: Optional[str] = "+0%") -> Tuple[Optional[bytes], Optional[str]]:
    voice = EDGE_VOICE_MAP.get(lang_code)
    if not voice:
        return None, f"No Edge voice for language '{lang_code}'."
    try:
        audio = asyncio.run(_edge_async(text, voice, rate))
        if not audio:
            return None, "Edge TTS returned no audio."
        return audio, None
    except Exception as e:
        return None, str(e)

def tts_gtts(text: str, lang_code: str) -> Tuple[Optional[bytes], Optional[str]]:
    if not gTTS:
        return None, "gTTS not available."
    try:
        buf = io.BytesIO()
        gTTS(text, lang=lang_code).write_to_fp(buf)
        return buf.getvalue(), None
    except Exception as e:
        return None, str(e)

def tts_gemini(text: str) -> Tuple[Optional[bytes], Optional[str]]:
    if not (genai and GEMINI_API_KEY):
        return None, "Gemini TTS not available."
    try:
        client = st.session_state.get("gemini_client") or genai.Client(api_key=GEMINI_API_KEY)
        cfg = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")
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

def synthesize_by_language(text: str, engine: str, lang_code: str) -> Tuple[Optional[bytes], str, Optional[str]]:
    """
    Returns (audio_bytes, mime, error or None)
    Falls back to gTTS if selected engine is unavailable for the chosen language.
    """
    # Primary engine
    if engine == "Edge-TTS":
        audio, err = tts_edge(text, lang_code, rate="+0%")
        if audio:
            return audio, "audio/mp3", None
        # Fallback to gTTS
        audio, err2 = tts_gtts(text, lang_code)
        return audio, "audio/mp3", err or err2
    elif engine == "Gemini TTS":
        audio, err = tts_gemini(text)
        if audio:
            return audio, "audio/mp3", None
        # Fallback to gTTS
        audio, err2 = tts_gtts(text, lang_code)
        return audio, "audio/mp3", err or err2
    else:  # gTTS
        audio, err = tts_gtts(text, lang_code)
        return audio, "audio/mp3", err

# --------------------------------
# RAG pipeline (with dynamic session context)
# --------------------------------
def rag_pipeline(query, selected_language):
    collection = get_collection()
    relevant = retrieve_documents(query)

    dynamic = ""
    if st.session_state.get("module_interaction_log"):
        lines = []
        for module, data in st.session_state.module_interaction_log.items():
            lines.append(f" - {module} (Last Run: {data.get('timestamp','')}) - Result: {data.get('result','')}")
        dynamic = "### SESSION HISTORY\n" + "\n".join(lines) + "\n\n"

    use_external = len(relevant) < 3 or get_collection().count() == 0
    if use_external:
        sys_inst = f"You are a friendly health consultant; use Google Search tool if needed. Reply in {selected_language}."
        prompt = dynamic + f"Answer the user's health question: {query}"
        r = call_gemini_api(prompt, system_instruction=sys_inst, tools=st.session_state.google_search_tool)
    else:
        kb = "\n".join(relevant)
        sys_inst = f"You are a medical assistant; use KB context and history. Reply in {selected_language}."
        prompt = dynamic + f"### KB\n{kb}\n\nQuestion: {query}\n\nAnswer:"
        r = call_gemini_api(prompt, system_instruction=sys_inst)

    if 'error' in r:
        return f"Generation error: {r['error']}"
    return r.get('response', "")

# --------------------------------
# Small utilities
# --------------------------------
def patient_input_form(key_prefix="p"):
    with st.form(key=f"form_{key_prefix}"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 0, 120, 45, key=f"{key_prefix}_age")
            gender = st.selectbox("Gender", ["Male","Female","Other"], key=f"{key_prefix}_gender")
            bmi = st.number_input("BMI", 10.0, 60.0, 25.0, key=f"{key_prefix}_bmi")
            sbp = st.number_input("Systolic BP", 60, 250, 120, key=f"{key_prefix}_sbp")
        with col2:
            dbp = st.number_input("Diastolic BP", 40, 160, 80, key=f"{key_prefix}_dbp")
            glucose = st.number_input("Glucose (mg/dL)", 40, 400, 100, key=f"{key_prefix}_glucose")
            cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 500, 180, key=f"{key_prefix}_cholesterol")
            smoker = st.selectbox("Smoker", ["No","Yes"], index=0, key=f"{key_prefix}_smoker")
        submitted = st.form_submit_button("Run Analysis")
    data = {"age":int(age),"gender":gender,"bmi":float(bmi),"sbp":float(sbp),
            "dbp":float(dbp),"glucose":float(glucose),"cholesterol":float(cholesterol),"smoker": smoker=="Yes"}
    return submitted, data

def preprocess_structured_input(data: Dict[str, Any]):
    arr = [data.get(k,0.0) for k in ["age","bmi","sbp","dbp","glucose","cholesterol"]]
    return np.array(arr, dtype=float).reshape(1,-1)

def sentiment_text(text: str, tok, model):
    if not tok or not model:
        return {"label":"unknown","score":0.0}
    inp = tok(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        out = model(**inp).logits
    p = torch.softmax(out, dim=-1).cpu().numpy()[0]
    idx = int(np.argmax(p)); labels = ["Negative","Neutral","Positive"]
    return {"label": labels[idx], "score": float(p[idx])}

# --------------------------------
# Sidebar
# --------------------------------
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
resp_mode = st.sidebar.selectbox("Response mode", ["Text","Voice"])
tts_engine = st.sidebar.selectbox("TTS engine", ["Gemini TTS","Edge-TTS","gTTS"])
answer_lang_name = st.sidebar.selectbox("Answer Language", list(LANGUAGE_DICT.keys()), index=0)
answer_lang_code = LANGUAGE_DICT[answer_lang_name]
st.session_state.selected_language = answer_lang_name

# Session vars
if 'module_interaction_log' not in st.session_state:
    st.session_state.module_interaction_log = {}

# Initialize for chat modules
if menu in ("🧠 RAG Chatbot","💡 Together Chat Assistant"):
    if 'db_client' not in st.session_state:
        st.session_state.db_client, st.session_state.model, st.session_state.gemini_client, st.session_state.google_search_tool = initialize_rag_dependencies()
        if get_collection().count() == 0:
            process_and_store_documents(split_documents(KB_TEXT))
            st.toast(f"Loaded {get_collection().count()} KB chunks.", icon="📚")

# Preload light models
text_tok, text_model = load_text_classifier()
sent_tok, sent_model = load_sentiment_model()
demo_clf, demo_reg = load_tabular_models()

# --------------------------------
# Modules (unchanged UI except chat TTS handling)
# --------------------------------
if menu == "💡 Together Chat Assistant":
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
                r = call_gemini_api(
                    prompt=prompt,
                    model_name="gemini-2.5-flash",
                    system_instruction="You are a helpful and medically accurate assistant. Keep it concise."
                )
                answer = r.get('response', r.get('error', "Unknown error."))
                st.write(answer)

                if resp_mode == "Voice":
                    audio, mime, err = synthesize_by_language(answer, tts_engine, answer_lang_code)
                    if audio:
                        st.audio(io.BytesIO(audio), format=mime)  # inline only
                    else:
                        st.warning(f"TTS failed: {err}")

                st.session_state.messages_assistant.append({"role": "assistant", "content": answer})

elif menu == "🧠 RAG Chatbot":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Knowledge Base")
    st.sidebar.info(f"KB Chunks: {get_collection().count()}")
    if st.sidebar.button("Reset/Reload Default KB"):
        try:
            st.session_state.db_client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass
        process_and_store_documents(split_documents(KB_TEXT))
        st.toast("KB reloaded.", icon="🔄")
        st.rerun()

    st.markdown("## Health RAG Chatbot 🧠 (Context-Aware)")
    if "messages_rag" not in st.session_state:
        st.session_state["messages_rag"] = [{"role":"assistant","content":"Hello! I'm your context-aware RAG medical assistant. Ask any health question."}]
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
                    audio, mime, err = synthesize_by_language(answer, tts_engine, answer_lang_code)
                    if audio:
                        st.audio(io.BytesIO(audio), format=mime)
                    else:
                        st.warning(f"TTS failed: {err}")
                st.session_state.messages_rag.append({"role":"assistant","content":answer})

# Keep your other analytical modules (Risk Stratification, LOS, Segmentation, etc.) as previously provided
# ...
