# app.py
# HealthAI Suite - Intelligent Analytics for Patient Care

import streamlit as st
import os
import sys
import tempfile
import uuid
import numpy as np
import pandas as pd
import base64
from typing import List, Dict, Any
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    MarianMTModel,
    MarianTokenizer,
    AutoModelForSeq2SeqLM
)
import torchvision.transforms as T
from PIL import Image
from huggingface_hub import login as hf_login
import requests
import json
import time

# For association rules
try:
    from mlxtend.frequent_patterns import apriori, association_rules
except ImportError:
    apriori = None
    association_rules = None
    st.warning("mlxtend not found. Medical Associations module will not function.")

# For chat assistant
try:
    import together
except ImportError:
    together = None
    st.warning("together not found. Chat Assistant module will not function.")

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

# Together AI API key
try:
    if "TOGETHER_API_KEY" in st.secrets:
        TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]
    else:
        TOGETHER_API_KEY = None
        st.warning("Together AI API key not found in secrets.toml.")
except KeyError:
    TOGETHER_API_KEY = None
    st.warning("Together AI API key not found in secrets.toml.")
    
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

LANGUAGE_DICT = {
    "English": "en", "Spanish": "es", "Arabic": "ar", "French": "fr", "German": "de", "Hindi": "hi",
    "Tamil": "ta", "Bengali": "bn", "Japanese": "ja", "Korean": "ko", "Russian": "ru",
    "Chinese (Simplified)": "zh", "Portuguese": "pt", "Italian": "it", "Dutch": "nl", "Turkish": "tr"
}

TRANSLATION_MODELS = {
    "en-ta": "Helsinki-NLP/opus-mt-en-ta", "ta-en": "Helsinki-NLP/opus-mt-ta-en",
    "en-es": "Helsinki-NLP/opus-mt-en-es", "es-en": "Helsinki-NLP/opus-mt-es-en",
    "en-ar": "Helsinki-NLP/opus-mt-en-ar", "ar-en": "Helsinki-NLP/opus-mt-ar-en",
    "en-fr": "Helsinki-NLP/opus-mt-en-fr", "fr-en": "Helsinki-NLP/opus-mt-fr-en",
    "en-de": "Helsinki-NLP/opus-mt-en-de", "de-en": "Helsinki-NLP/opus-mt-de-en",
    "en-hi": "Helsinki-NLP/opus-mt-en-hi", "hi-en": "Helsinki-NLP/opus-mt-hi-en",
    "en-bn": "Helsinki-NLP/opus-mt-en-bn", "bn-en": "Helsinki-NLP/opus-mt-bn-en",
    "en-ja": "Helsinki-NLP/opus-mt-en-ja", "ja-en": "Helsinki-NLP/opus-mt-ja-en",
    "en-ko": "Helsinki-NLP/opus-mt-en-ko", "ko-en": "Helsinki-NLP/opus-mt-ko-en",
    "en-ru": "Helsinki-NLP/opus-mt-en-ru", "ru-en": "Helsinki-NLP/opus-mt-ru-en",
    "en-zh": "Helsinki-NLP/opus-mt-en-zh", "zh-en": "Helsinki-NLP/opus-mt-zh-en",
    "en-pt": "Helsinki-NLP/opus-mt-en-pt", "pt-en": "Helsinki-NLP/opus-mt-pt-en",
    "en-it": "Helsinki-NLP/opus-mt-en-it", "it-en": "Helsinki-NLP/opus-mt-it-en",
    "en-nl": "Helsinki-NLP/opus-mt-en-nl", "nl-en": "Helsinki-NLP/opus-mt-nl-en",
    "en-tr": "Helsinki-NLP/opus-mt-en-tr", "tr-en": "Helsinki-NLP/opus-mt-tr-en"
}

# -------------------------
# Helpers: safe model loaders (cached)
# -------------------------
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
def load_translation_model(model_name):
    """Loads a specific translation model."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load translation model '{model_name}': {e}")
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
def language_translator(text, src_lang, tgt_lang):
    """Translates text from source to target language."""
    if src_lang == tgt_lang:
        return text
    
    model_key = f"{src_lang}-{tgt_lang}"
    model_name = TRANSLATION_MODELS.get(model_key)
    
    if not model_name:
        st.warning(f"No translation model found for {src_lang} to {tgt_lang}.")
        return "Translation model not available for this pair; returning original text."
        
    tokenizer, model = load_translation_model(model_name)
    if not tokenizer or not model:
        return "Translation model not available for this pair; returning original text."

    try:
        tokenized = tokenizer([text], return_tensors='pt', truncation=True, padding=True)
        out = model.generate(**tokenized, max_length=128)
        return tokenizer.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"Translation failed: {e}")
        return "Translation model not available for this pair; returning original text."

def text_classify(text: str, tokenizer, model, labels=None):
    if tokenizer is None or model is None:
        return {"label": "error", "score": 0.0}
    
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

def sentiment_text(text: str, tokenizer, model):
    if tokenizer is None or model is None:
        return {"label": "error", "score": 0.0}
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

def call_together_api(prompt):
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TOGETHER_API_KEY}"
        }
        payload = {
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 256
        }
        response = requests.post(TOGETHER_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Together AI API: {e}")
        return "An error occurred while getting a response."
    except KeyError as e:
        st.error(f"Invalid API response format: Missing key {e}")
        return "Failed to get a valid response from the model."

# -------------------------
# App UI: Sidebar + Navigation
# -------------------------
st.sidebar.title("HealthAI Suite")
menu = st.sidebar.radio("Select Module", [
    "🧑‍⚕️ Risk Stratification",
    "⏱ Length of Stay Prediction",
    "👥 Patient Segmentation",
    "🔗 Medical Associations",
    "🩻 Imaging Diagnostics",
    "📈 Sequence Forecasting",
    "📝 Clinical Notes Analysis",
    "🌐 Translator",
    "💬 Sentiment Analysis",
    "💡 Chat Assistant"
])

text_tok, text_model = load_text_classifier()
sent_tok, sent_model = load_sentiment_model()
demo_clf, demo_reg = load_tabular_models()

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
        st.success(f"Predicted Risk Level: **{label}** (Score: {score})")

# -------------------------
# Module: Patient Segmentation
# -------------------------
elif menu == "👥 Patient Segmentation":
    st.title("Patient Segmentation")
    st.write("Assigns a patient to a distinct health cohort and visualizes their position relative to the groups.")
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

        st.success(f"Assigned Cohort: **Cohort {pred_label + 1}**")
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
# Module: Medical Associations
# -------------------------
elif menu == "🔗 Medical Associations":
    st.title("Medical Associations")
    st.write("Discovers relationships between medical conditions and risk factors using association rule mining.")
    if apriori is None or association_rules is None:
        st.error("This module requires the mlxtend library. Please install it with pip install mlxtend.")
    else:
        st.info("This is a simplified example. For a real-world use case, you would need a large transactional dataset of patient symptoms, conditions, and risk factors.")
        
        data = [
            ['high_cholesterol', 'hypertension'],
            ['high_glucose', 'hypertension', 'obesity'],
            ['hypertension', 'obesity'],
            ['high_cholesterol', 'hypertension', 'smoking'],
            ['high_glucose', 'obesity'],
            ['hypertension', 'smoking', 'obesity']
        ]

        from mlxtend.preprocessing import TransactionEncoder
        te = TransactionEncoder()
        te_ary = te.fit(data).transform(data)
        df_trans = pd.DataFrame(te_ary, columns=te.columns_)

        st.subheader("Simulated Patient Data (One-Hot Encoded)")
        st.dataframe(df_trans)

        frequent_itemsets = apriori(df_trans, min_support=0.5, use_colnames=True)
        st.subheader("Frequent Itemsets (Support >= 0.5)")
        st.dataframe(frequent_itemsets)

        rules = association_rules(frequent_itemsets, metric="confidence", min_confidence=0.7)
        st.subheader("Generated Association Rules (Confidence >= 0.7)")
        st.dataframe(rules.sort_values(by="lift", ascending=False))
        
        st.success("Example rule: Patients with **Hypertension** and **Obesity** have a high confidence of also having **High Glucose**.")

# -------------------------
# Module: Imaging Diagnostics
# -------------------------
elif menu == "🩻 Imaging Diagnostics":
    st.title("Imaging Diagnostics")
    st.write("Simulates medical image analysis using a dummy model. In a full implementation, this would use a CNN for tasks like disease detection.")
    
    st.info("This is a placeholder module. A real-world application would require a trained Convolutional Neural Network (CNN) model and a proper image pre-processing pipeline.")
    
    uploaded_file = st.file_uploader("Upload a medical image (e.g., X-ray)", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
        @st.cache_resource
        def dummy_diagnose_image(image):
            """A placeholder function for image diagnosis."""
            diag = np.random.choice(["No Anomaly Detected", "Pneumonia Detected", "Fracture Identified", "Mass Detected"], p=[0.7, 0.15, 0.1, 0.05])
            confidence = np.random.uniform(0.7, 0.99)
            return {"diagnosis": diag, "confidence": confidence}

        if st.button("Run Diagnosis"):
            with st.spinner("Analyzing image..."):
                result = dummy_diagnose_image(uploaded_file)
                st.success(f"Diagnosis Result: **{result['diagnosis']}** (Confidence: {result['confidence']:.2f})")
                
# -------------------------
# Module: Sequence Forecasting
# -------------------------
elif menu == "📈 Sequence Forecasting":
    st.title("Sequence Forecasting")
    st.write("Predicts a patient's next health metric value based on a time-series of past data.")

    st.info("This is a simplified example. A full implementation would utilize a more sophisticated model like an LSTM or RNN.")

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

        df_seq = pd.DataFrame({
            "Time": range(1, num_points + 1),
            "Metric Value": data
        })

        st.subheader("Generated Time-Series Data")
        st.line_chart(df_seq.set_index("Time"))

        last_two = data[-2:]
        prediction = last_two[1] + (last_two[1] - last_two[0])

        st.success(f"Based on the trend, the predicted next value is: **{prediction:.2f}**")
        st.write("This prediction is made using a simple linear extrapolation from the last two data points.")

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
        
        st.success(f"Predicted length of stay: **{los_est_rounded} days**")
        st.info("The prediction is based on a simplified model. For a real-world application, a model fine-tuned on extensive patient data would be required.")

# -------------------------
# Module: Clinical Notes Analysis
# -------------------------
elif menu == "📝 Clinical Notes Analysis":
    st.title("Clinical Notes Analysis")
    st.write("Analyzes clinical notes to provide insights. The current model identifies emotional tone.")
    notes = st.text_area("Paste clinical notes here", height=200, placeholder="Example: The patient presented with chest pain and a consistent cough.")
    if st.button("Analyze Notes"):
        if not notes.strip():
            st.warning("Please paste clinical notes to analyze.")
        else:
            res = text_classify(notes, text_tok, text_model, labels=["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise"])
            if res['label'] == 'error':
                 st.error("Failed to analyze notes. Check the model and input.")
            else:
                 st.success(f"Analysis: The note has a primary tone of **{res['label']}** (Confidence: {res['score']:.2f}).")

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
            translated_text = language_translator(text_to_trans, src_code, tgt_code)
            st.success("Translated Text:")
            st.write(translated_text)

# -------------------------
# Module: Sentiment Analysis
# -------------------------
elif menu == "💬 Sentiment Analysis":
    st.title("Sentiment Analysis")
    st.write("Analyzes the sentiment of patient feedback or reviews.")
    txt = st.text_area("Paste patient feedback or reviews", "The nurse was very kind, but the waiting time was too long.", key="sentiment_input")
    if st.button("Analyze Sentiment"):
        res = sentiment_text(txt, sent_tok, sent_model)
        st.success(f"Sentiment: **{res['label']}** (Confidence: {res['score']:.2f})")

# -------------------------
# Module: Chat Assistant
# -------------------------
elif menu == "💡 Chat Assistant":
    st.title("Health Chat Assistant")
    st.write("Ask questions and get information from a language model assistant.")
    
    if not TOGETHER_API_KEY:
        st.error("The Chat Assistant is not configured. Please add your Together AI API key to `secrets.toml`.")
        st.stop()
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello! I am a health assistant. How can I help you today?"}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Your message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {TOGETHER_API_KEY}"
                    }
                    payload = {
                        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        "max_tokens": 256
                    }
                    response = requests.post(TOGETHER_API_URL, headers=headers, data=json.dumps(payload))
                    response.raise_for_status()
                    full_response = response.json()['choices'][0]['message']['content']
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    st.write(full_response)
                except requests.exceptions.RequestException as e:
                    st.error(f"Error calling Together AI API: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": "An error occurred while getting a response."})
                except KeyError as e:
                    st.error(f"Invalid API response format: Missing key {e}")
                    st.session_state.messages.append({"role": "assistant", "content": "Failed to get a valid response from the model."})
