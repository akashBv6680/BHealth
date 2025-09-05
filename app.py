# app.py
# HealthAI Suite - Intelligent Analytics for Patient Care

import streamlit as st
import os
import sys
import tempfile
import uuid
import json
import requests
import time
from datetime import datetime
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import torch
import torchvision.transforms as T
from PIL import Image
from huggingface_hub import login as hf_login

# This block MUST be at the very top to fix the sqlite3 version issue for ChromaDB.
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules['pysqlite3']
except ImportError:
    pass

# Now import chromadb and other libraries for the RAG component
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import Hugging Face models
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    MarianMTModel,
    MarianTokenizer,
)

# For other models
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# For association rules (optional)
try:
    from mlxtend.frequent_patterns import apriori, association_rules
except ImportError:
    apriori = None
    association_rules = None
    st.warning("mlxtend not found. Medical Associations module will not function.")

# For Together AI chat assistant (optional)
try:
    import together
except ImportError:
    together = None
    st.warning("together not found. Together Chat Assistant module will not function.")

# -------------------------
# App config
# -------------------------
st.set_page_config(page_title="HealthAI Suite", page_icon="🩺", layout="wide")

# Hugging Face login
try:
    if "HF_ACCESS_TOKEN" in st.secrets:
        hf_login(token=st.secrets["HF_ACCESS_TOKEN"], add_to_git_credential=False)
    else:
        st.warning("Hugging Face access token not found in secrets.toml.")
except Exception as e:
    st.error(f"Hugging Face login failed: {e}")

# Together AI config
TOGETHER_API_KEY = st.secrets.get("TOGETHER_API_KEY")
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

# Dictionary of supported languages and their ISO 639-1 codes
LANGUAGE_DICT = {
    "English": "en", "Spanish": "es", "Arabic": "ar", "French": "fr", "German": "de", "Hindi": "hi",
    "Tamil": "ta", "Bengali": "bn", "Japanese": "ja", "Korean": "ko", "Russian": "ru",
    "Chinese (Simplified)": "zh-Hans", "Portuguese": "pt", "Italian": "it", "Dutch": "nl", "Turkish": "tr"
}

# -------------------------
# RAG Chatbot specific configurations
# -------------------------
COLLECTION_NAME = "rag_documents"

# --- Placeholder Knowledge Base for the Chatbot ---
# This has been expanded to include more comprehensive medical information
KNOWLEDGE_BASE_TEXT = """
### Common Cold
The common cold is a viral infection of your nose and throat (upper respiratory tract). It's usually harmless, although it might not feel that way. Many types of viruses can cause a common cold. Symptoms include a runny or stuffy nose, sore throat, cough, congestion, and sneezing. Rest, staying hydrated with fluids, and using over-the-counter medications like pain relievers and nasal decongestants are key for recovery. Cold symptoms typically last 7 to 10 days.

### Diabetes (Type 2)
Type 2 diabetes is a chronic condition that affects the way your body processes blood sugar (glucose). Your body either doesn't produce enough insulin, or it resists insulin. This can lead to high blood sugar levels. Symptoms can include increased thirst and urination, fatigue, and blurry vision. Management involves a combination of a healthy diet, regular exercise, and medications such as Metformin or insulin therapy to control blood sugar levels. Regular monitoring of blood glucose is essential.

### Hypertension (High Blood Pressure)
Hypertension is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. It's often called the "silent killer" because it may not have obvious symptoms. It can be caused by factors like high sodium intake, lack of exercise, and genetics. Regular blood pressure monitoring, a low-sodium diet (DASH diet is recommended), regular physical activity, and prescribed medications like ACE inhibitors or diuretics are crucial for control.

### Migraine
A migraine is a type of headache that can cause severe throbbing pain or a pulsing sensation, usually on one side of the head. It's often accompanied by nausea, vomiting, and extreme sensitivity to light and sound. Migraine attacks can last from a few hours to several days. Triggers can include certain foods, stress, and changes in sleep patterns. Treatments include acute pain relief medications (triptans, NSAIDs) and preventative drugs (beta-blockers, anti-seizure medications) to reduce the frequency and severity of attacks.

### Asthma
Asthma is a chronic condition in which your airways narrow and swell and may produce extra mucus. This can make breathing difficult and trigger coughing, a whistling sound (wheezing) when you breathe out, and shortness of breath. For some people, asthma is a minor nuisance. For others, it can be a major problem that interferes with daily activities. Inhalers (relievers and preventers) are a primary form of treatment. Reliever inhalers (like albuterol) are used for quick relief during an attack, while preventer inhalers (corticosteroids) are used daily to reduce airway inflammation.

### Depression
Depression is a mood disorder that causes a persistent feeling of sadness and loss of interest. Also called major depressive disorder, it affects how you feel, think, and behave and can lead to a variety of emotional and physical problems. Symptoms include feeling sad, hopeless, or empty, changes in appetite or sleep, and loss of energy. It is caused by a combination of genetic, biological, environmental, and psychological factors. Treatment options include psychotherapy (like cognitive-behavioral therapy or CBT), antidepressant medications (SSRIs, SNRIs), and lifestyle changes.

### Anxiety Disorders
Anxiety disorders are a group of mental health disorders characterized by feelings of anxiety and fear. Unlike the brief anxiety you might feel before a presentation, an anxiety disorder involves intense, excessive, and persistent worry and fear about everyday situations. Common types include generalized anxiety disorder (GAD), panic disorder, and social anxiety disorder. Symptoms can include restlessness, difficulty concentrating, and physical symptoms like a rapid heartbeat. Treatment often involves a combination of psychotherapy, medications, and stress management techniques.

### Osteoarthritis
Osteoarthritis is the most common form of arthritis, affecting millions of people worldwide. It occurs when the protective cartilage on the ends of your bones wears down over time. Although osteoarthritis can damage any joint, the disorder most commonly affects joints in your hands, knees, hips, and spine. Symptoms include pain, stiffness, and loss of flexibility. Treatment focuses on managing pain and improving joint function. This can include exercise, weight management, physical therapy, and medications like pain relievers and anti-inflammatory drugs. In some severe cases, surgery may be considered.
"""

# -------------------------
# Helpers: safe model loaders (cached)
# -------------------------
@st.cache_resource(show_spinner=False)
def initialize_rag_dependencies():
    """Initializes ChromaDB client and SentenceTransformer model for RAG."""
    try:
        db_path = tempfile.mkdtemp()
        db_client = chromadb.PersistentClient(path=db_path)
        # Explicitly load the model to the CPU
        model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        return db_client, model
    except Exception as e:
        st.error(f"An error occurred during RAG dependency initialization: {e}.")
        st.stop()

@st.cache_resource(show_spinner=False)
def load_text_classifier(model_name="bhadresh-savani/bert-base-uncased-emotion"):
    """Loads a text classification model."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load text classifier model: {e}")
        return None, None

@st.cache_resource(show_spinner=False)
def load_translation_model(src_lang="en", tgt_lang="hi"):
    """Loads MarianMT translation model for src->tgt."""
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
    """Loads a sentiment analysis model."""
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        m = AutoModelForSequenceClassification.from_pretrained(model_name)
        m.eval()
        return tok, m
    except Exception:
        return None, None

@st.cache_resource(show_spinner=False)
def load_tabular_models():
    """Loads light scikit-learn models for demo."""
    clf = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=50, random_state=42))])
    reg = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestRegressor(n_estimators=50, random_state=42))])
    return clf, reg

# -------------------------
# Utility functions
# -------------------------
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
    except Exception as e:
        st.error(f"Error during text classification: {e}")
        return {"label": "error", "score": 0.0}

def translate_text(text: str, src: str, tgt: str):
    tkn, m = load_translation_model(src, tgt)
    if tkn is None or m is None:
        return "Translation model not available for this pair; returning original text."
    
    inputs = tkn.prepare_seq2seq_batch([text], return_tensors="pt")
    with torch.no_grad():
        translated = m.generate(**inputs)
    out = tkn.batch_decode(translated, skip_special_tokens=True)[0]
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
# RAG functions
# -------------------------
def get_collection():
    """Retrieves or creates the ChromaDB collection."""
    return st.session_state.db_client.get_or_create_collection(
        name=COLLECTION_NAME
    )

def call_together_api(prompt, max_retries=5):
    """Calls the Together AI API with exponential backoff for retries."""
    if not TOGETHER_API_KEY:
        st.error("Together AI API key is not configured.")
        return {"error": "API Key not found."}
    
    retry_delay = 1
    for i in range(max_retries):
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {TOGETHER_API_KEY}"
            }
            payload = {
                "model": "mistralai/Mistral-7B-Instruct-v0.2",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 1024
            }
            response = requests.post(TOGETHER_API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                st.warning(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            elif e.response.status_code == 401:
                st.error("Invalid API Key. Please check your Together AI API key.")
                return {"error": "401 Unauthorized"}
            else:
                st.error(f"Failed to call API after {i+1} retries: {e}")
                return {"error": str(e)}
        except Exception as e:
            st.error(f"An error occurred during the API call: {e}")
            return {"error": str(e)}

def clear_chroma_data():
    """Clears all data from the ChromaDB collection."""
    try:
        if COLLECTION_NAME in [col.name for col in st.session_state.db_client.list_collections()]:
            st.session_state.db_client.delete_collection(name=COLLECTION_NAME)
    except Exception as e:
        st.error(f"Error clearing collection: {e}")

def split_documents(text_data, chunk_size=500, chunk_overlap=100):
    """Splits a single string of text into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_text(text_data)

def is_valid_github_raw_url(url):
    """Checks if a URL is a valid GitHub raw file URL."""
    pattern = r"https://raw\.githubusercontent\.com/[\w-]+/[\w-]+/[^/]+/[\w./-]+\.(txt|md)"
    return re.match(pattern, url) is not None

def process_and_store_documents(documents):
    """
    Processes a list of text documents, generates embeddings, and
    stores them in ChromaDB.
    """
    collection = get_collection()
    model = st.session_state.model

    embeddings = model.encode(documents).tolist()
    document_ids = [str(uuid.uuid4()) for _ in documents]
    
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=document_ids
    )

    st.toast("Documents processed and stored successfully!", icon="✅")

def retrieve_documents(query, n_results=5):
    """
    Retrieves the most relevant documents from ChromaDB based on a query.
    """
    collection = get_collection()
    model = st.session_state.model
    
    query_embedding = model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    return results['documents'][0]

def rag_pipeline(query, selected_language_code):
    """Executes the full RAG pipeline with a check for documents."""
    collection = get_collection()
    if collection.count() == 0:
        return "Hello! I'm a chatbot that answers questions based on a knowledge base. Please add documents before asking me anything. I'm ready when you are! 😊"

    relevant_docs = retrieve_documents(query)
    
    context = "\n".join(relevant_docs)
    
    # Translate the user's query before sending to the LLM
    translated_query = translate_text(query, LANGUAGE_DICT['English'], selected_language_code)
    
    prompt = f"Using the following information, answer the user's question. The final response MUST be in {st.session_state.selected_language}. If the information is not present, state that you cannot answer. \n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    response_json = call_together_api(prompt)

    if 'error' in response_json:
        return "An error occurred while generating the response. Please try again."
    
    try:
        response_text = response_json['choices'][0]['message']['content']
        # The LLM's response is already in the target language due to the prompt.
        return response_text
    except (KeyError, IndexError):
        st.error("Invalid API response format.")
        return "Failed to get a valid response from the model."

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

# Initialize shared resources
text_tok, text_model = load_text_classifier()
sent_tok, sent_model = load_sentiment_model()
demo_clf, demo_reg = load_tabular_models()
if menu == "🧠 RAG Chatbot":
    if 'db_client' not in st.session_state or 'model' not in st.session_state:
        st.session_state.db_client, st.session_state.model = initialize_rag_dependencies()
        # Automatically load the default knowledge base
        with st.spinner("Loading and processing default knowledge base..."):
            documents = split_documents(KNOWLEDGE_BASE_TEXT)
            process_and_store_documents(documents)

# -------------------------
# Common patient form fields (used across pages)
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

# -------------------------
# Module: Risk Stratification
# -------------------------
if menu == "🧑‍⚕️ Risk Stratification":
    st.title("Risk Stratification")
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

# -------------------------
# Other modules follow...
# ... (all other modules from the original script) ...
# -------------------------

# -------------------------
# Module: Length of Stay Prediction
# -------------------------
elif menu == "⏱ Length of Stay Prediction":
    st.title("Length of Stay Prediction")
    st.write("Predicts the expected hospital length of stay (in days) for a patient.")
    submitted, pdata = patient_input_form("los")
    if submitted:
        los_est = 3.0 + (pdata['age']/30.0) + (pdata['bmi']/40.0) + (pdata['glucose']/200.0)
        los_est_rounded = int(round(los_est))
        st.success(f"Predicted length of stay: *{los_est_rounded} days*")
        st.info("The prediction is based on a simplified model.")

# -------------------------
# Module: Patient Segmentation
# -------------------------
elif menu == "👥 Patient Segmentation":
    st.title("Patient Segmentation")
    st.write("Assigns a patient to a distinct health cohort.")
    submitted, pdata = patient_input_form("seg")
    if submitted:
        X_new = preprocess_structured_input(pdata)
        rng = np.random.RandomState(42)
        synthetic_data = rng.normal(loc=[50,25,120,80,100,180], scale=[15,5,20,10,30,40], size=(200,6))
        X_all = np.vstack([synthetic_data, X_new])
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_all)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Xs)
        pred_label = kmeans.predict(Xs[-1].reshape(1, -1))[0]
        st.success(f"Assigned Cohort: *Cohort {pred_label + 1}*")
        st.write("The patient's profile is most similar to Cohort " + str(pred_label + 1) + ".")
        st.subheader("Patient's Position within Cohorts")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(Xs)
        df_vis = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
        df_vis['Cohort'] = kmeans.labels_
        df_vis['Cohort'] = df_vis['Cohort'].astype(str)
        df_vis.loc[len(df_vis)-1, 'Cohort'] = 'New Patient'
        fig, ax = plt.subplots(figsize=(8, 6))
        cohort_colors = {0: 'blue', 1: 'green', 2: 'purple', 'New Patient': 'red'}
        for cohort_num in range(kmeans.n_clusters):
            subset = df_vis[df_vis['Cohort'] == str(cohort_num)]
            ax.scatter(subset['PCA1'], subset['PCA2'], alpha=0.7, label=f'Cohort {cohort_num+1}', color=cohort_colors[cohort_num])
        new_patient_point = df_vis[df_vis['Cohort'] == 'New Patient']
        ax.scatter(new_patient_point['PCA1'], new_patient_point['PCA2'], marker='*', s=300, label='New Patient', color=cohort_colors['New Patient'], edgecolor='black')
        ax.set_title("Patient Cohorts (2D PCA Visualization)")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.legend()
        st.pyplot(fig)
        st.subheader("Cohort Characteristics")
        cols = ["Age", "BMI", "SBP", "DBP", "Glucose", "Cholesterol"]
        df_avg = pd.DataFrame(columns=cols)
        for cohort_num in range(kmeans.n_clusters):
            cluster_indices = np.where(kmeans.labels_ == cohort_num)[0]
            avg_vals = np.mean(X_all[cluster_indices], axis=0)
            df_avg.loc[f"Cohort {cohort_num+1}"] = avg_vals
        st.dataframe(df_avg.style.format("{:.2f}"))
        st.write("This table shows the average values for each key metric in each cohort.")

# -------------------------
# Module: Imaging Diagnostics
# -------------------------
elif menu == "🩻 Imaging Diagnostics":
    st.title("Imaging Diagnostics")
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

# -------------------------
# Module: Sequence Forecasting
# -------------------------
elif menu == "📈 Sequence Forecasting":
    st.title("Sequence Forecasting")
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
        st.write("This prediction is made using a simple linear extrapolation.")

# -------------------------
# Module: Clinical Notes Analysis
# -------------------------
elif menu == "📝 Clinical Notes Analysis":
    st.title("Clinical Notes Analysis")
    st.write("Analyzes clinical notes to provide insights.")
    notes = st.text_area("Paste clinical notes here", height=200, placeholder="Example: The patient presented with chest pain and a consistent cough.")
    if st.button("Analyze Notes"):
        if not notes.strip():
            st.warning("Please paste clinical notes to analyze.")
        else:
            res = text_classify(notes, text_tok, text_model, labels=["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise"])
            if res['label'] == 'error':
                 st.error("Failed to analyze notes.")
            else:
                 st.success(f"Analysis: The note has a primary tone of *{res['label']}* (Confidence: {res['score']:.2f}).")

# -------------------------
# Module: Translator
# -------------------------
elif menu == "🌐 Translator":
    st.title("Translator")
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

# -------------------------
# Module: Sentiment Analysis
# -------------------------
elif menu == "💬 Sentiment Analysis":
    st.title("Patient Feedback Sentiment Analysis")
    st.write("Analyzes patient feedback to determine the sentiment.")
    patient_feedback = st.text_area("Patient Feedback", "The nurse was very helpful, but the wait time was too long.", key="sentiment_input")
    if st.button("Analyze Sentiment"):
        if not patient_feedback.strip():
            st.warning("Please provide some feedback to analyze.")
        else:
            sentiment_result = sentiment_text(patient_feedback, sent_tok, sent_model)
            if sentiment_result['label'] == 'unknown':
                st.error("Sentiment analysis model could not be loaded. Check your dependencies.")
            else:
                st.success(f"Sentiment: **{sentiment_result['label']}** (Confidence: {sentiment_result['score']:.2f})")

# -------------------------
# Module: Together Chat Assistant
# -------------------------
elif menu == "💡 Together Chat Assistant":
    st.title("Together AI Chat Assistant")
    st.write("Ask questions and get information from a language model assistant.")
    
    if not TOGETHER_API_KEY:
        st.error("Together AI API key is not configured. Please add it to `secrets.toml`.")
    else:
        try:
            import together
            together.api_key = TOGETHER_API_KEY
        except Exception as e:
            st.error(f"Together AI library initialization failed: {e}")
            together = None

    if "messages_together" not in st.session_state:
        st.session_state["messages_together"] = [
            {"role": "assistant", "content": "Hello! I am a general health assistant. How can I help you today?"}
        ]

    for msg in st.session_state.messages_together:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask me anything about general health..."):
        if not together:
            st.chat_message("assistant").write("The chat assistant is not configured.")
            st.stop()
        
        st.session_state.messages_together.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    chat_completion = together.Complete.create(
                        prompt=prompt,
                        model="mistralai/Mixtral-8x7B-Instruct-v0.1"
                    )
                    full_response = chat_completion['choices'][0]['text']
                    st.session_state.messages_together.append({"role": "assistant", "content": full_response})
                    st.write(full_response)
                except Exception as e:
                    st.error(f"Chatbot failed: {e}")

# -------------------------
# NEW Module: RAG Chatbot
# -------------------------
elif menu == "🧠 RAG Chatbot":
    st.title("Health RAG Chatbot")
    st.write("Ask questions about specific medical conditions. This chatbot is augmented with a knowledge base.")

    # Sidebar for RAG configuration and history
    with st.sidebar:
        st.markdown("---")
        st.subheader("RAG Settings")
        st.session_state.selected_language = st.selectbox(
            "Select a Language for response",
            options=list(LANGUAGE_DICT.keys()),
            key="rag_language_selector"
        )
        if st.button("New Chat", key="rag_new_chat"):
            st.session_state.messages_rag = []
            clear_chroma_data()
            st.session_state.chat_history = {}
            st.session_state.current_chat_id = None
            st.experimental_rerun()
        
    # Initialize RAG chatbot state
    if 'messages_rag' not in st.session_state:
        st.session_state.messages_rag = []
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = {}
    if 'current_chat_id' not in st.session_state:
        st.session_state.current_chat_id = str(uuid.uuid4())
        st.session_state.chat_history[st.session_state.current_chat_id] = {
            'messages': st.session_state.messages_rag,
            'title': "New Chat",
            'date': datetime.now()
        }

    # Display chat messages
    for message in st.session_state.messages_rag:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask about the health conditions..."):
        st.session_state.messages_rag.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                selected_language_code = LANGUAGE_DICT[st.session_state.selected_language]
                response = rag_pipeline(prompt, selected_language_code)
                st.markdown(response)

        st.session_state.messages_rag.append({"role": "assistant", "content": response})
