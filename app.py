# app.py
# HealthAI Suite - Intelligent Analytics for Patient Care

import streamlit as st
import os
import sys
import tempfile
import uuid
import time
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px  # Added for Patient Segmentation
from typing import List, Dict, Any
import torch
import torchvision.transforms as T
from PIL import Image
import datetime  # Added for the logging timestamps

# This block MUST be at the very top to fix the sqlite3 version issue for ChromaDB.
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules['pysqlite3']
except ImportError:
    pass

# Now import RAG libraries
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import Hugging Face models (placeholders)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    MarianMTModel,
    MarianTokenizer,
)

# For other models
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA  # Added for Patient Segmentation

# For Gemini AI chat assistant and RAG
try:
    from google import genai
    from google.genai.errors import APIError
    from google.genai import types
except ImportError:
    genai = None
    APIError = None
    types = None

# -------------------------
# App config
# -------------------------
st.set_page_config(page_title="HealthAI Suite", page_icon="🩺", layout="wide")

# Gemini AI config
# NOTE: Ensure GEMINI_API_KEY is set in your Streamlit secrets (secrets.toml)
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

# Dictionary of supported languages and their ISO 639-1 codes (MULTILINGUAL - NO CHANGE)
LANGUAGE_DICT = {
    "English": "en", "Spanish": "es", "Arabic": "ar", "French": "fr", "German": "de", "Hindi": "hi",
    "Tamil": "ta", "Bengali": "bn", "Japanese": "ja", "Korean": "ko", "Russian": "ru",
    "Chinese (Simplified)": "zh-Hans", "Portuguese": "pt", "Italian": "it", "Dutch": "nl", "Turkish": "tr"
}

# -------------------------
# RAG Chatbot specific configurations
# -------------------------
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
# Helpers: RAG/Gemini Initialization (Cached Resources)
# -------------------------
@st.cache_resource(show_spinner=False)
def initialize_rag_dependencies():
    """Initializes ChromaDB client, SentenceTransformer model, and Gemini client."""
    try:
        # Use a temporary directory for ChromaDB storage
        db_path = tempfile.mkdtemp()
        db_client = chromadb.PersistentClient(path=db_path)

        # Load embedding model
        model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')

        # Initialize Gemini Client and configure Google Search Tool
        if GEMINI_API_KEY and genai:
            gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            google_search_tool = [types.Tool(google_search={})]
        else:
            gemini_client = None
            google_search_tool = None
            if GEMINI_API_KEY is None:
                st.error("GEMINI_API_KEY is missing from Streamlit secrets.")

        return db_client, model, gemini_client, google_search_tool
    except Exception as e:
        st.error(f"An error occurred during RAG dependency initialization: {e}. Check dependencies and API Key.")
        st.stop()

# --- Placeholder Model Loaders ---
@st.cache_resource(show_spinner=False)
def load_text_classifier(model_name="bhadresh-savani/bert-base-uncased-emotion"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        return tokenizer, model
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
# RAG/Gemini core functions
# -------------------------
def get_collection():
    """Retrieves or creates the ChromaDB collection."""
    if 'db_client' not in st.session_state:
        st.session_state.db_client, st.session_state.model, st.session_state.gemini_client, st.session_state.google_search_tool = initialize_rag_dependencies()
    return st.session_state.db_client.get_or_create_collection(
        name=COLLECTION_NAME
    )

def clear_and_reload_kb():
    """Clears the existing collection and reloads the default KB."""
    if 'db_client' not in st.session_state:
        st.session_state.db_client, _, _, _ = initialize_rag_dependencies()

    db_client = st.session_state.db_client

    # 1. Delete the existing collection
    try:
        db_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass  # Ignore if collection doesn't exist

    # 2. Process and store the default documents
    with st.spinner("Processing new knowledge base..."):
        documents = split_documents(KNOWLEDGE_BASE_TEXT)
        process_and_store_documents(documents)

    # Reset chat history and interaction log
    kb_count = get_collection().count()
    st.session_state["messages_rag"] = [
        {"role": "assistant", "content": f"Hello! I'm your RAG medical assistant. The Knowledge Base has been reset and now contains **{kb_count}** chunks, including information on the HealthAI Suite modules. Ask me anything!"}
    ]
    st.session_state.module_interaction_log = {}  # Reset the dynamic log
    st.rerun()

def call_gemini_api(prompt, model_name="gemini-2.5-flash", system_instruction="You are a helpful health assistant.", tools=None, max_retries=5):
    """Calls the Gemini API with optional tools and exponential backoff for retries."""
    if not st.session_state.get('gemini_client'):
        return {"error": "API Client not found. Check GEMINI_API_KEY in secrets.toml."}

    retry_delay = 1
    for i in range(max_retries):
        try:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=tools
            )

            response = st.session_state.gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )

            return {"response": response.text}

        except APIError as e:
            if i < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            return {"error": str(e)}
        except Exception as e:
            return {"error": str(e)}

def split_documents(text_data, chunk_size=500, chunk_overlap=100):
    """Splits a single string of text into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_text(text_data)

def process_and_store_documents(documents):
    """Stores documents in ChromaDB."""
    collection = get_collection()
    model = st.session_state.model

    embeddings = model.encode(documents).tolist()
    document_ids = [str(uuid.uuid4()) for _ in documents]

    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=document_ids
    )

def retrieve_documents(query, n_results=5):
    """Retrieves relevant documents from ChromaDB."""
    collection = get_collection()
    model = st.session_state.model

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=['documents', 'distances']
    )

    return results['documents'][0] if results['documents'] else []

# *** CRITICAL FUNCTION: RAG PIPELINE WITH DYNAMIC CONTEXT INJECTION ***
def rag_pipeline(query, selected_language):
    """
    Executes the RAG pipeline with dynamic session context and LLM fallback.
    """
    collection = get_collection()
    relevant_docs = retrieve_documents(query)

    # 1. GENERATE DYNAMIC SESSION CONTEXT
    dynamic_context = ""
    if st.session_state.get("module_interaction_log"):
        log_entries = []
        for module, data in st.session_state.module_interaction_log.items():
            # Precompute each line safely
            ts = data.get('timestamp', '')
            res = data.get('result', '')
            log_entries.append(f" - {module} (Last Run: {ts}) - Result: {res}")

        # Avoid f-string backslash expression; compute joined text first
        joined_entries = "\n".join(log_entries)

        dynamic_context = (
            "### CURRENT USER SESSION HISTORY (CRITICAL CONTEXT)\n"
            "The user recently performed the following analysis/interactions within the HealthAI Suite:\n"
            + joined_entries + "\n"
            "You MUST use this history if it is relevant to the user's current question, otherwise ignore it.\n\n"
        )

    # 2. DETERMINE RAG STRATEGY (KB or External Search)
    use_external_search = len(relevant_docs) < 3 or collection.count() == 0

    if use_external_search:
        # --- External Search (Consultant Mode with Citations) ---
        system_instruction = (
            f"You are a friendly, helpful, and highly knowledgeable health consultant. "
            f"You must use the Google Search tool to find reliable, current health information "
            f"to answer the user's query: '{query}'. "
            f"Your response must be comprehensive, easy to understand, and provide advice as a good consultant would. "
            f"You MUST cite all external sources used with numbered links at the end of your response. "
            f"The final response MUST be in {selected_language}."
        )

        # Inject dynamic context into the prompt
        fallback_query = (
            f"{dynamic_context}"
            f"Provide a detailed, consultant-style answer to the user's health question: {query}. "
            "Ensure all facts are supported by your search results or the provided user history."
        )

        response_json = call_gemini_api(
            prompt=fallback_query,
            system_instruction=system_instruction,
            tools=st.session_state.google_search_tool
        )

    else:
        # --- Internal RAG (KB Context + Dynamic Context) ---
        kb_context = "\n".join(relevant_docs)

        rag_system_instruction = (
            "You are a medical assistant and health consultant. Use the provided context AND the user's session history to answer the question. "
            "Your answer should be detailed and clarifying. "
            "If neither the static context nor the history contains the answer, you MUST politely state, "
            "'I apologize, but my specific knowledge base does not contain information to answer that question.' "
            f"The final response MUST be in {selected_language}"
        )

        prompt = (
            f"{dynamic_context}"
            f"### STATIC KNOWLEDGE BASE CONTEXT\n{kb_context}\n\n"
            f"Question: {query}\n\nAnswer:"
        )

        response_json = call_gemini_api(prompt, system_instruction=rag_system_instruction)

    if 'error' in response_json:
        return f"An error occurred while generating the response: {response_json['error']}."

    try:
        return response_json['response']
    except KeyError:
        return "Failed to get a valid response from the model."

# -------------------------
# Utility Functions
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
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        if labels:
            lbl = labels[pred]
        else:
            lbl = str(pred)
        return {"label": lbl, "score": float(probs[pred])}
    except Exception:
        return {"label": "error", "score": 0.0}

def translate_text(text: str, src: str, tgt: str):
    tkn, m = load_translation_model(src, tgt)
    if tkn is None or m is None:
        return "Translation model not available for this pair; returning original text."
    # Newer Transformers recommend direct tokenizer call rather than prepare_seq2seq_batch
    inputs = tkn([text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        generated = m.generate(**inputs)
    out = tkn.batch_decode(generated, skip_special_tokens=True)[0]
    return out

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
# App UI: Sidebar + Navigation
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

# Initialize session state variables
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = "English"
# Initialized for dynamic context
if 'module_interaction_log' not in st.session_state:
    st.session_state.module_interaction_log = {}

# Initialization block for RAG/Gemini dependencies and KB loading
if menu == "🧠 RAG Chatbot" or menu == "💡 Together Chat Assistant":
    if 'db_client' not in st.session_state:
        st.session_state.db_client, st.session_state.model, st.session_state.gemini_client, st.session_state.google_search_tool = initialize_rag_dependencies()

        if st.session_state.db_client and get_collection().count() == 0:
            with st.spinner("Loading and processing default knowledge base..."):
                documents = split_documents(KNOWLEDGE_BASE_TEXT)
                process_and_store_documents(documents)
                st.toast(f"Loaded {get_collection().count()} KB chunks.", icon="📚")

# Load other non-LLM resources
text_tok, text_model = load_text_classifier()
sent_tok, sent_model = load_sentiment_model()
demo_clf, demo_reg = load_tabular_models()

# -------------------------
# Module: Risk Stratification
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
        st.success(f"Predicted Risk Level: *{label}* (Score: {score})")

        # *** DYNAMIC CONTEXT UPDATE ***
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        result_str = f"Predicted Risk Level: {label} (Score: {score}). Input patient data: Age {pdata['age']}, BMI {pdata['bmi']}, Glucose {pdata['glucose']}."
        st.session_state.module_interaction_log[menu] = {"timestamp": current_time, "result": result_str}

# -------------------------
# Module: Length of Stay Prediction
# -------------------------
elif menu == "⏱ Length of Stay Prediction":
    st.title("Length of Stay Prediction ⏱")
    st.write("Predicts the expected hospital length of stay (in days) for a patient.")
    submitted, pdata = patient_input_form("los")
    if submitted:
        # Simplified prediction logic
        los_est = 3.0 + (pdata['age']/30.0) + (pdata['bmi']/40.0) + (pdata['glucose']/200.0)
        los_est_rounded = int(round(los_est))
        st.success(f"Predicted length of stay: *{los_est_rounded} days*")

        # *** DYNAMIC CONTEXT UPDATE ***
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        result_str = f"Predicted LOS: {los_est_rounded} days. Input patient data: Age {pdata['age']}, Glucose {pdata['glucose']}."
        st.session_state.module_interaction_log[menu] = {"timestamp": current_time, "result": result_str}

# -------------------------
# Module: Patient Segmentation (WITH VISUALIZATION - CONSOLIDATED & CORRECT)
# -------------------------
elif menu == "👥 Patient Segmentation":
    st.title("Patient Segmentation 👥")
    st.write("Assigns a patient to a distinct health cohort and visualizes the clustering using 3D PCA.")
    submitted, pdata = patient_input_form("seg")
    if submitted:
        # 1. Prepare Data
        X_new = preprocess_structured_input(pdata)
        rng = np.random.RandomState(42)
        # 6 features: age, bmi, sbp, dbp, glucose, cholesterol
        synthetic_data = rng.normal(loc=[50,25,120,80,100,180], scale=[15,5,20,10,30,40], size=(200,6))
        X_all = np.vstack([synthetic_data, X_new])

        # 2. Scale and Cluster
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_all)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Xs[:-1])  # Fit on synthetic data only

        # Predict label for new patient and all data
        pred_label_index = kmeans.predict(Xs[-1].reshape(1, -1))[0]
        all_labels = kmeans.predict(Xs)
        cohort_label = f"Cohort {pred_label_index + 1}"

        st.success(f"Assigned Cohort: **{cohort_label}**")
        st.info("The visualization below shows the 6-dimensional data reduced to 3 dimensions (PCA) for plotting. Your patient is the red dot.")

        # 3. Visualization using PCA (Reduce 6D to 3D)
        pca = PCA(n_components=3, random_state=42)
        X_pca = pca.fit_transform(Xs)

        # Create DataFrame for Plotly
        df_plot = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2', 'PCA3'])
        df_plot['Cohort'] = [f"Cohort {l+1}" for l in all_labels]
        df_plot['Type'] = ['Existing' for _ in range(len(synthetic_data))] + ['New Patient']

        # Build figure; include 'Type' and 'Cohort' in custom_data for hover
        fig = px.scatter_3d(
            df_plot,
            x='PCA1',
            y='PCA2',
            z='PCA3',
            color='Type',  # Color by 'Type' to highlight 'New Patient' in red
            symbol='Type',
            opacity=0.8,
            title="Patient Segmentation (PCA-Reduced to 3D)",
            color_discrete_map={'Existing': 'blue', 'New Patient': 'red'},
            custom_data=['Type', 'Cohort']
        )

        fig.update_traces(
            marker=dict(size=[10 if t == 'New Patient' else 5 for t in df_plot['Type']]),
            hovertemplate="<b>Type:</b> %{customdata[0]}<br><b>Cohort:</b> %{customdata[1]}<br>PCA1: %{x}<br>PCA2: %{y}<br>PCA3: %{z}<extra></extra>"
        )

        st.plotly_chart(fig, use_container_width=True)

        # *** DYNAMIC CONTEXT UPDATE ***
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        result_str = f"Patient assigned to {cohort_label}. K-Means clustering performed on 6 health metrics and visualized via 3D PCA."
        st.session_state.module_interaction_log[menu] = {"timestamp": current_time, "result": result_str}

# -------------------------
# Module: Imaging Diagnostics
# -------------------------
elif menu == "🩻 Imaging Diagnostics":
    st.title("Imaging Diagnostics 🩻")
    st.write("Simulates medical image analysis using a dummy model.")
    st.info("This is a placeholder module.")
    uploaded_file = st.file_uploader("Upload a medical image (e.g., X-ray)", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        @st.cache_resource
        def dummy_diagnose_image(image):
            diag = np.random.choice(["No Anomaly Detected", "Pneumonia Detected", "Fracture Identified", "Mass Detected"], p=[0.7, 0.15, 0.1, 0.05])
            confidence = np.random.uniform(0.7, 0.99)
            return {"diagnosis": diag, "confidence": confidence}

        if st.button("Run Diagnosis"):
            with st.spinner("Analyzing image..."):
                result = dummy_diagnose_image(uploaded_file)
                st.success(f"Diagnosis Result: *{result['diagnosis']}* (Confidence: {result['confidence']:.2f})")

                # *** DYNAMIC CONTEXT UPDATE ***
                current_time = datetime.datetime.now().strftime("%H:%M:%S")
                result_str = f"Dummy image diagnosis: {result['diagnosis']} (Confidence: {result['confidence']:.2f})."
                st.session_state.module_interaction_log[menu] = {"timestamp": current_time, "result": result_str}

# -------------------------
# Module: Sequence Forecasting
# -------------------------
elif menu == "📈 Sequence Forecasting":
    st.title("Sequence Forecasting 📈")
    st.write("Predicts a patient's next health metric value based on a time-series of past data.")
    st.info("This is a simplified example.")
    col1, col2 = st.columns(2)
    with col1:
        num_points = st.slider("Number of data points to generate", 5, 50, 15)
    with col2:
        noise_level = st.slider("Noise level", 0.0, 1.0, 0.1)
    if st.button("Generate Data and Predict"):
        np.random.seed(42)
        trend = np.linspace(50, 80, num_points)
        noise = np.random.normal(0, noise_level * 10, num_points)
        data = trend + noise
        df_seq = pd.DataFrame({"Time": range(1, num_points + 1), "Metric Value": data})
        st.subheader("Generated Time-Series Data")
        st.line_chart(df_seq.set_index("Time"))
        last_two = data[-2:]
        prediction = last_two[1] + (last_two[1] - last_two[0])
        st.success(f"Based on the trend, the predicted next value is: *{prediction:.2f}*")

        # *** DYNAMIC CONTEXT UPDATE ***
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        result_str = f"Predicted next value: {prediction:.2f} using {num_points} data points (noise: {noise_level})."
        st.session_state.module_interaction_log[menu] = {"timestamp": current_time, "result": result_str}

# -------------------------
# Module: Clinical Notes Analysis
# -------------------------
elif menu == "📝 Clinical Notes Analysis":
    st.title("Clinical Notes Analysis 📝")
    st.write("Analyzes clinical notes to provide insights.")
    notes = st.text_area("Paste clinical notes here", height=200, placeholder="Example: The patient presented with chest pain and a consistent cough.")
    if st.button("Analyze Notes"):
        if not notes.strip():
            st.warning("Please paste clinical notes to analyze.")
        else:
            res = text_classify(notes, text_tok, text_model, labels=["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise"])
            if res['label'] in ('error', 'unknown'):
                st.error("Failed to analyze notes. Check model loading.")
                analysis_result_desc = "Analysis failed."
            else:
                analysis_result_desc = f"Tone: {res['label']} (Confidence: {res['score']:.2f})"
                st.success(f"Analysis: The note has a primary tone of *{res['label']}* (Confidence: {res['score']:.2f}).")

            # *** DYNAMIC CONTEXT UPDATE ***
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            snippet = (notes[:30] + "...") if len(notes) > 30 else notes
            result_str = f"Clinical Note Analysis performed. {analysis_result_desc} Note snippet: '{snippet}'."
            st.session_state.module_interaction_log[menu] = {"timestamp": current_time, "result": result_str}

# -------------------------
# Module: Translator
# -------------------------
elif menu == "🌐 Translator":
    st.title("Translator 🌐")
    st.write("Translate clinical or patient-facing text between different languages.")
    col1, col2 = st.columns(2)
    with col1:
        src_lang = st.selectbox("Source Language", list(LANGUAGE_DICT.keys()), index=0)
    with col2:
        tgt_lang = st.selectbox("Target Language", list(LANGUAGE_DICT.keys()), index=1)

    text_to_trans = st.text_area("Text to translate", "Please describe your symptoms and any medications you are taking.", key="translator_input")

    if st.button("Translate"):
        src_code = LANGUAGE_DICT.get(src_lang, "en")
        tgt_code = LANGUAGE_DICT.get(tgt_lang, "en")

        with st.spinner("Translating..."):
            translated_text = translate_text(text_to_trans, src_code, tgt_code)
            st.success("Translated Text:")
            st.write(translated_text)

            # *** DYNAMIC CONTEXT UPDATE ***
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            snippet = (translated_text[:30] + "...") if len(translated_text) > 30 else translated_text
            result_str = f"Text translated from {src_lang} to {tgt_lang}. Translation snippet: '{snippet}'."
            st.session_state.module_interaction_log[menu] = {"timestamp": current_time, "result": result_str}

# -------------------------
# Module: Sentiment Analysis
# -------------------------
elif menu == "💬 Sentiment Analysis":
    st.title("Patient Feedback Sentiment Analysis 💬")
    st.write("Analyzes patient feedback to determine the sentiment.")
    patient_feedback = st.text_area("Patient Feedback", "The nurse was very helpful, but the wait time was too long.", key="sentiment_input")
    if st.button("Analyze Sentiment"):
        if not patient_feedback.strip():
            st.warning("Please provide some feedback to analyze.")
        else:
            sentiment_result = sentiment_text(patient_feedback, sent_tok, sent_model)
            if sentiment_result['label'] == 'unknown':
                st.error("Sentiment analysis model could not be loaded. Check your dependencies.")
                sentiment_label = "Analysis failed."
                confidence = 0.0
            else:
                sentiment_label = sentiment_result['label']
                confidence = sentiment_result['score']
                st.success(f"Sentiment: **{sentiment_label}** (Confidence: {confidence:.2f})")

            # *** DYNAMIC CONTEXT UPDATE ***
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            snippet = (patient_feedback[:30] + "...") if len(patient_feedback) > 30 else patient_feedback
            result_str = f"Sentiment analyzed: {sentiment_label} (Confidence: {confidence:.2f}). Feedback snippet: '{snippet}'."
            st.session_state.module_interaction_log[menu] = {"timestamp": current_time, "result": result_str}

# -------------------------
# Module: Together Chat Assistant (General Purpose LLM)
# -------------------------
elif menu == "💡 Together Chat Assistant":
    st.title("Together Chat Assistant 💡 (Powered by Gemini)")
    st.write("Ask questions and get general health information from a language model assistant.")

    if not GEMINI_API_KEY:
        st.error("The chat assistant is not configured. Please check your GEMINI_API_KEY in `secrets.toml`.")
        st.stop()

    if "messages_assistant" not in st.session_state:
        st.session_state["messages_assistant"] = [
            {"role": "assistant", "content": "Hello! I am a general health assistant powered by Google Gemini. How can I help you today?"}
        ]

    for msg in st.session_state.messages_assistant:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask me anything about general health..."):
        if not st.session_state.get('gemini_client'):
            st.chat_message("assistant").write("The chat assistant is not configured. Please check your API key.")
            st.stop()

        st.session_state.messages_assistant.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_json = call_gemini_api(
                    prompt=prompt,
                    model_name="gemini-2.5-flash",
                    system_instruction="You are a helpful and medically accurate general health assistant. Keep your answers concise."
                )
                full_response = response_json.get('response', response_json.get('error', "An unknown error occurred."))
                st.write(full_response)
                st.session_state.messages_assistant.append({"role": "assistant", "content": full_response})

# -------------------------
# Module: RAG Chatbot (Context-Aware)
# -------------------------
elif menu == "🧠 RAG Chatbot":

    # --- RAG Settings in Sidebar (Language Selection & KB Management) ---
    st.sidebar.markdown("---")
    st.sidebar.header("RAG Settings")

    # Language selection for RAG output (MULTILINGUAL - NO CHANGE)
    st.session_state.selected_language = st.sidebar.selectbox(
        "Select Response Language",
        list(LANGUAGE_DICT.keys()),
        index=0,
        key="rag_lang_select_sidebar"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Knowledge Base Management")

    # Display current KB status (Checked after load attempt)
    kb_count = get_collection().count()
    st.sidebar.info(f"Knowledge Base Chunks: **{kb_count}**")

    # Button to reset and reload KB
    if st.sidebar.button("Reset/Reload Default KB (Updated)", key="reset_kb_button"):
        clear_and_reload_kb()
        st.toast("Knowledge Base reset and reloaded with default health and module data!", icon="🔄")

    # --- Main Chat Interface ---
    st.markdown("## Health RAG Chatbot 🧠 (Context-Aware)")
    st.markdown("A specialized **Health Consultant AI**. It uses its fixed knowledge base (KB), falls back to **Google Search** for current info, AND is aware of your **recent activity** in the other modules to provide contextual replies.")

    # Initialize RAG chat history
    if "messages_rag" not in st.session_state:
        st.session_state["messages_rag"] = [
            {"role": "assistant", "content": "Hello! I'm your context-aware RAG medical assistant. I know the capabilities of all our modules. Try running an analysis (e.g., Risk Stratification) and then ask me about the result!"}
        ]

    # Display chat messages
    for msg in st.session_state.messages_rag:
        st.chat_message(msg["role"]).write(msg["content"])

    # Handle user input
    if prompt := st.chat_input("Ask a health question or about your recent activity..."):
        if not st.session_state.get('gemini_client'):
            st.chat_message("assistant").write("The RAG chatbot is not configured. Please check your Gemini API key.")
            st.stop()

        st.session_state.messages_rag.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context, checking history, and generating response..."):
                # Call the RAG pipeline with the LLM/Google Search fallback logic
                full_response = rag_pipeline(st.session_state.messages_rag[-1]["content"], st.session_state.selected_language)

                st.write(full_response)

                st.session_state.messages_rag.append({"role": "assistant", "content": full_response})
