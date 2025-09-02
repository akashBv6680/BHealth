# app.py
# HealthAI Suite - Intelligent Analytics for Patient Care

import streamlit as st
import os
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
    MarianTokenizer
)
import torchvision.transforms as T
from PIL import Image
from huggingface_hub import login as hf_login

# For association rules
try:
    from mlxtend.frequent_patterns import apriori, association_rules
except Exception:
    apriori = None
    association_rules = None

# For chat assistant
try:
    import together
except ImportError:
    together = None

# -------------------------
# App config
# -------------------------
st.set_page_config(page_title="HealthAI Suite", page_icon="🩺", layout="wide")

try:
    hf_login(token=st.secrets["HF_ACCESS_TOKEN"], add_to_git_credential=False)
except KeyError:
    st.warning("Hugging Face access token not found. Some models may not load.")
except Exception as e:
    st.error(f"Hugging Face login failed: {e}")

LANGUAGE_DICT = {
    "English": "en", "Spanish": "es", "Arabic": "ar", "French": "fr", "German": "de", "Hindi": "hi",
    "Tamil": "ta", "Bengali": "bn", "Japanese": "ja", "Korean": "ko", "Russian": "ru",
    "Chinese (Simplified)": "zh-Hans", "Portuguese": "pt", "Italian": "it", "Dutch": "nl", "Turkish": "tr"
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

def translate_text(text: str, src: str, tgt: str):
    tkn, m = load_translation_model(src, tgt)
    if tkn is None or m is None:
        return "Translation service is currently unavailable for this language pair."
    inputs = tkn.prepare_seq2seq_batch([text], return_tensors="pt")
    with torch.no_grad():
        translated = m.generate(**{k: v for k, v in inputs.items()})
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
# App UI: Sidebar + Navigation
# -------------------------
st.sidebar.title("HealthAI Suite")
menu = st.sidebar.radio("Select Module", [
    "🧑‍⚕️ Risk Stratification",
    "⏱ Length of Stay Prediction",
    "👥 Patient Segmentation",
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

---

### Patient Segmentation Improvement

The original Patient Segmentation module provided a single cluster number, which isn't very informative. The improved version below:

* **Generates a Scatter Plot:** It visualizes the patient's position relative to the discovered cohorts, providing a clear, graphical understanding of the segmentation.
* **Shows Cluster Characteristics:** It displays the average values for each cohort, allowing for easy comparison and interpretation of what makes each group unique. For example, you can see that "Cohort 3" might have a higher average BMI and cholesterol.

This makes the results much more intuitive for a client or user.

---

```python
# -------------------------
# Module: Patient Segmentation
# -------------------------
elif menu == "👥 Patient Segmentation":
    st.title("Patient Segmentation")
    st.write("Assigns a patient to a distinct health cohort and visualizes their position relative to the groups.")
    submitted, pdata = patient_input_form("seg")
    if submitted:
        X_new = preprocess_structured_input(pdata)
        
        # Generate synthetic data for clustering
        rng = np.random.RandomState(42)
        synthetic_data = rng.normal(loc=[50,25,120,80,100,180], scale=[15,5,20,10,30,40], size=(200,6))
        
        # Combine new patient data with synthetic data for clustering
        X_all = np.vstack([synthetic_data, X_new])
        
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_all)
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Xs)
        pred_label = kmeans.predict(Xs[-1].reshape(1, -1))[0]
        
        # --- Visualization ---
        st.success(f"Assigned Cohort: **Cohort {pred_label + 1}**")
        st.write("The patient's profile is most similar to Cohort " + str(pred_label + 1) + ".")
        
        st.subheader("Patient's Position within Cohorts")
        
        # Use PCA for 2D visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(Xs)
        
        df_vis = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
        df_vis['Cohort'] = kmeans.labels_
        df_vis['Cohort'] = df_vis['Cohort'].astype(str)
        df_vis.loc[len(df_vis)-1, 'Cohort'] = 'New Patient'
        
        fig, ax = plt.subplots(figsize=(8, 6))
        cohort_colors = {0: 'blue', 1: 'green', 2: 'purple', 'New Patient': 'red'}
        
        # Plot each cohort
        for cohort_num in range(kmeans.n_clusters):
            subset = df_vis[df_vis['Cohort'] == str(cohort_num)]
            ax.scatter(subset['PCA1'], subset['PCA2'], alpha=0.7, label=f'Cohort {cohort_num+1}', color=cohort_colors[cohort_num])
            
        # Plot the new patient
        new_patient_point = df_vis[df_vis['Cohort'] == 'New Patient']
        ax.scatter(new_patient_point['PCA1'], new_patient_point['PCA2'], marker='*', s=300, label='New Patient', color=cohort_colors['New Patient'], edgecolor='black')
        
        ax.set_title("Patient Cohorts (2D PCA Visualization)")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.legend()
        st.pyplot(fig)
        
        # --- Cohort Characteristics ---
        st.subheader("Cohort Characteristics")
        
        # Create a DataFrame for average values of each cohort
        cols = ["Age", "BMI", "SBP", "DBP", "Glucose", "Cholesterol"]
        df_avg = pd.DataFrame(columns=cols)
        
        for cohort_num in range(kmeans.n_clusters):
            cluster_indices = np.where(kmeans.labels_ == cohort_num)[0]
            avg_vals = np.mean(X_all[cluster_indices], axis=0)
            df_avg.loc[f"Cohort {cohort_num+1}"] = avg_vals
        
        st.dataframe(df_avg.style.format("{:.2f}"))
        st.write("This table shows the average values for each key metric in each cohort.")

---

### Other Module Improvements

* **Length of Stay Prediction:** The message "This is a demo estimate" has been replaced with a more professional statement indicating the model's status.
* **Clinical Notes Analysis:** The module is now renamed from "Clinical NLP" to "Clinical Notes Analysis" for clearer understanding. The analysis message is also simplified to show a clear outcome based on the model.
* **Translator:** The code now includes error handling to show a user-friendly message if the translation fails, rather than a generic error. The core of the issue with the translator not working properly is often due to an inability to connect to the Hugging Face model repository or an invalid language pair, both of which are now handled more gracefully.

Here are the remaining sections of the updated code.

---

```python
# -------------------------
# Module: Length of Stay Prediction
# -------------------------
elif menu == "⏱ Length of Stay Prediction":
    st.title("Length of Stay Prediction")
    st.write("Predicts the expected hospital length of stay (in days) for a patient.")
    submitted, pdata = patient_input_form("los")
    if submitted:
        los_est = 3.0 + (pdata['age']/30.0) + (pdata['bmi']/40.0) + (pdata['glucose']/200.0)
        los_est = round(float(los_est), 2)
        st.success(f"Predicted length of stay: **{los_est} days**")
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
            translated_text = translate_text(text_to_trans, src_code, tgt_code)
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
    
    try:
        together.api_key = st.secrets["TOGETHER_API_KEY"]
    except KeyError:
        st.error("Together API key not found in secrets.toml.")
        together = None

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello! I am a health assistant. How can I help you today?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        if not together:
            st.chat_message("assistant").write("The chat assistant is not configured.")
            st.stop()
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    chat_completion = together.Complete.create(
                        prompt=prompt,
                        model="mistralai/Mixtral-8x7B-Instruct-v0.1"
                    )
                    full_response = chat_completion['choices'][0]['text']
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    st.write(full_response)
                except Exception as e:
                    st.error(f"Chatbot failed: {e}")
