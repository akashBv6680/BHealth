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

# Import Hugging Face models (placeholders for other modules)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    MarianMTModel,
    MarianTokenizer,
)

# For other models (placeholders for other modules)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# For Gemini AI chat assistant and RAG
try:
    from google import genai
    from google.genai.errors import APIError
    from google.genai import types
except ImportError:
    genai = None
    APIError = None
    types = None
    # If the user runs this without the Google GenAI SDK, the chatbot modules will display an error.


# -------------------------
# App config
# -------------------------
st.set_page_config(page_title="HealthAI Suite", page_icon="🩺", layout="wide")

# The Hugging Face login block is often unnecessary for Streamlit demos unless using private models
# try:
#     if "HF_ACCESS_TOKEN" in st.secrets:
#         from huggingface_hub import login as hf_login
#         hf_login(token=st.secrets["HF_ACCESS_TOKEN"], add_to_git_credential=False)
# except Exception:
#     pass

# Gemini AI config
# NOTE: Ensure GEMINI_API_KEY is set in your Streamlit secrets (secrets.toml)
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

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

# --- MASSIVELY EXPANDED Placeholder Knowledge Base for the Chatbot ---
KNOWLEDGE_BASE_TEXT = """
### Common Cold and Flu
The common cold is a viral infection of your upper respiratory tract. Symptoms include a runny/stuffy nose, sore throat, cough, and congestion. Rest, hydration, and OTC medications are key. The flu (influenza) is more severe, often causing fever, body aches, and extreme fatigue. Antiviral drugs may be used for the flu. Cold symptoms last 7-10 days, while the flu can last longer and pose a higher risk of complications.

### Diabetes (Type 2) Management
Type 2 diabetes is a chronic condition characterized by high blood sugar due to insulin resistance or insufficient insulin production. Management involves: a healthy diet, regular exercise (at least 150 minutes of moderate activity per week), blood glucose monitoring, and medications like Metformin. Uncontrolled diabetes can lead to heart disease, nerve damage, and kidney failure.

### Hypertension (High Blood Pressure)
Hypertension is defined as chronically elevated blood pressure, often symptomless (the 'silent killer'). Risk factors include high sodium intake, lack of physical activity, and genetics. Treatment involves lifestyle changes (DASH diet, exercise) and medications (ACE inhibitors, diuretics, beta-blockers). Normal blood pressure is typically below 120/80 mmHg.

### Migraine and Headache
A migraine is a neurological condition causing severe, throbbing headaches, often with nausea and light/sound sensitivity. Triggers include stress, certain foods (aged cheese, wine), and sleep changes. Acute treatment uses triptans or NSAIDs; prevention involves daily medications like beta-blockers. Tension headaches are the most common type, causing mild to moderate pain.

### Asthma Control
Asthma is a chronic condition where airways narrow and swell. Symptoms include wheezing, shortness of breath, and coughing. Treatment relies on two main types of inhalers: **Reliever** (quick-relief, like albuterol, used during an attack) and **Preventer** (daily inhaled corticosteroids, used to reduce inflammation). A written Asthma Action Plan is crucial.

### Mental Health: Depression and Anxiety
**Depression** is a persistent feeling of sadness and loss of interest. Treatment includes psychotherapy (e.g., CBT) and antidepressant medications (SSRIs). **Anxiety disorders** involve excessive worry and fear; treatment also includes therapy, medication, and mindfulness techniques. Seeking help from a mental health professional is vital.

### Nutrition and Diet
A balanced diet is essential for cardiovascular health. Key components include: high intake of fruits, vegetables, and whole grains; lean proteins; healthy fats (avocados, nuts, olive oil); and limited consumption of processed foods, added sugars, and saturated/trans fats. **The Mediterranean Diet** is widely recommended for long-term health.

### Exercise Guidelines
Adults should aim for at least **150 minutes of moderate-intensity** aerobic exercise (like brisk walking or swimming) or **75 minutes of vigorous-intensity** exercise per week. Strength training should be done for all major muscle groups at least two days per week. Regular exercise lowers the risk of heart disease, diabetes, and certain cancers.

### Common Skin Conditions
**Eczema** (dermatitis) causes dry, itchy, inflamed patches of skin; treated with moisturizers and topical steroids. **Acne** involves clogged pores, often treated with benzoyl peroxide, retinoids, or oral antibiotics. Sun protection is key to preventing skin cancer and premature aging.

### Basic First Aid
For minor cuts, clean the area with soap and water, apply an antiseptic, and cover with a sterile bandage. For minor burns, cool the area immediately with cold running water for at least 10 minutes. Seek emergency care for deep cuts, large burns, or signs of infection.

### Heart Health
Key indicators of a healthy heart include a resting heart rate between 60-100 beats per minute, normal blood pressure (below 120/80 mmHg), and healthy cholesterol levels (low LDL, high HDL). Lifestyle factors like quitting smoking and regular cardiovascular exercise are the most effective preventive measures against heart disease.
"""

# -------------------------
# Helpers: safe model loaders (cached)
# -------------------------
@st.cache_resource(show_spinner=False)
def initialize_rag_dependencies():
    """Initializes ChromaDB client, SentenceTransformer model, and Gemini client."""
    try:
        db_path = tempfile.mkdtemp()
        db_client = chromadb.PersistentClient(path=db_path)
        # Using a fast and effective embedding model
        model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        
        # Initialize Gemini Client and configure Google Search Tool
        if GEMINI_API_KEY and genai:
            gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            # Correct way to define the search tool for the API config
            google_search_tool = [types.Tool(google_search={})] 
        else:
            gemini_client = None
            google_search_tool = None
            if GEMINI_API_KEY is None:
                 st.error("GEMINI_API_KEY is missing from Streamlit secrets.")

        return db_client, model, gemini_client, google_search_tool
    except Exception as e:
        st.error(f"An error occurred during RAG dependency initialization: {e}. Check dependencies (pysqlite3, chromadb, sentence-transformers, google-genai).")
        st.stop()


# Placeholder function for other modules (kept for completeness)
def load_text_classifier(model_name="bhadresh-savani/bert-base-uncased-emotion"):
    # Simplified return for demo completeness
    return None, None

def load_tabular_models():
    # Simplified return for demo completeness
    clf = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=50, random_state=42))])
    reg = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestRegressor(n_estimators=50, random_state=42))])
    return clf, reg

def preprocess_structured_input(pdata):
    # Simplified return for demo completeness
    gender_val = 1 if pdata['gender'] == 'Male' else (0 if pdata['gender'] == 'Female' else 0.5)
    smoker_val = 1 if pdata['smoker'] else 0
    return np.array([pdata['age'], pdata['bmi'], pdata['sbp'], pdata['dbp'], pdata['glucose'], pdata['cholesterol']])

def text_classify(notes, tokenizer, model, labels):
    # Simplified return for demo completeness
    return {'label': 'Neutral', 'score': 0.85}

def translate_text(text, src_code, tgt_code):
    # Simplified return for demo completeness
    return f"Translation of '{text}' from {src_code} to {tgt_code} is not available in this demo."

def sentiment_text(feedback, tokenizer, model):
    # Simplified return for demo completeness
    return {'label': 'positive', 'score': 0.9}


# -------------------------
# RAG/Gemini core functions
# -------------------------
def get_collection():
    """Retrieves or creates the ChromaDB collection, ensuring dependencies are initialized."""
    if 'db_client' not in st.session_state:
         # Initialize dependencies if they haven't been yet
         st.session_state.db_client, st.session_state.model, st.session_state.gemini_client, st.session_state.google_search_tool = initialize_rag_dependencies()
    return st.session_state.db_client.get_or_create_collection(
        name=COLLECTION_NAME
    )

def clear_and_reload_kb():
    """Clears the existing collection and reloads the default KB."""
    if 'db_client' not in st.session_state:
        # Should already be initialized, but ensure safety
        st.session_state.db_client, _, _, _ = initialize_rag_dependencies()
        
    db_client = st.session_state.db_client
    
    # 1. Delete the existing collection
    try:
        db_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        # Ignore if collection doesn't exist
        pass
    
    # 2. Process and store the default documents
    with st.spinner("Processing new knowledge base..."):
        documents = split_documents(KNOWLEDGE_BASE_TEXT)
        process_and_store_documents(documents)
    
    # Reset chat history to reflect the KB change
    kb_count = get_collection().count()
    st.session_state["messages_rag"] = [
        {"role": "assistant", "content": f"Hello! I'm your RAG medical assistant. The Knowledge Base has been reset and now contains **{kb_count}** chunks. Ask me about common health topics!"}
    ]
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
            if "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(retry_delay)
                retry_delay *= 2
            elif "API_KEY_INVALID" in str(e):
                return {"error": "401 Unauthorized: Invalid API Key"}
            else:
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

def retrieve_documents(query, n_results=5):
    """
    Retrieves the most relevant documents from ChromaDB based on a query.
    """
    collection = get_collection()
    model = st.session_state.model
    
    query_embedding = model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=['documents', 'distances']
    )
    
    return results['documents'][0] if results['documents'] else []

def rag_pipeline(query, selected_language):
    """
    Executes the RAG pipeline with an LLM fallback to Google Search tool for OOKB queries.
    """
    collection = get_collection()
    
    # --- RAG Step 1: Attempt Retrieval ---
    relevant_docs = retrieve_documents(query)
    
    # --- RAG Step 2: Context Check and Fallback Logic ---
    # Use external search if retrieval is poor (fewer than 3 docs) OR if KB is empty
    if len(relevant_docs) < 3 or collection.count() == 0: 
        
        # 1. Fallback to Google Search Tool (External Knowledge/Consultant)
        system_instruction = (
            f"You are a friendly, helpful, and highly knowledgeable health consultant. "
            f"You must use the Google Search tool to find reliable, current health information "
            f"to answer the user's query: '{query}'. "
            f"Your response must be comprehensive, easy to understand, and answer all parts of the user's question, providing advice as a good consultant would. "
            f"You MUST cite all external sources used with numbered links at the end of your response. "
            f"The final response MUST be in {selected_language}."
        )
        
        fallback_query = f"Provide a detailed, consultant-style answer to the user's health question: {query}. Ensure all facts are supported by your search results."
        
        response_json = call_gemini_api(
            prompt=fallback_query, 
            system_instruction=system_instruction,
            tools=st.session_state.google_search_tool # Use the configured search tool
        )
        
    else:
        # --- RAG Success Step (Use KB context) ---
        
        context = "\n".join(relevant_docs)
        
        rag_system_instruction = (
            "You are a medical assistant and health consultant. Use ONLY the provided context to answer the user's question. "
            "Your answer should be detailed and clarifying, acting as a good consultant. "
            "If the context does not contain the answer, you MUST politely state, "
            "'I apologize, but my specific knowledge base does not contain information to answer that question.' "
            f"The final response MUST be in {selected_language}"
        )
        
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        response_json = call_gemini_api(prompt, system_instruction=rag_system_instruction)

    if 'error' in response_json:
        return f"An error occurred while generating the response: {response_json['error']}. Please check your API key and try again."
    
    try:
        response_text = response_json['response']
        return response_text
    except KeyError:
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
    "💡 Gemini Chat Assistant",
    "🧠 RAG Chatbot"
])

# Initialize session state for language selection
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = "English"

# Initialization block for RAG/Gemini dependencies and KB loading
if menu == "🧠 RAG Chatbot" or menu == "💡 Gemini Chat Assistant":
    if 'db_client' not in st.session_state:
        # Load all RAG/Gemini dependencies
        st.session_state.db_client, st.session_state.model, st.session_state.gemini_client, st.session_state.google_search_tool = initialize_rag_dependencies()
        
        # Load the default knowledge base if it's the RAG Chatbot AND the KB is empty
        if menu == "🧠 RAG Chatbot":
            if st.session_state.db_client and get_collection().count() == 0:
                with st.spinner("Loading and processing default knowledge base (large)..."):
                    documents = split_documents(KNOWLEDGE_BASE_TEXT)
                    process_and_store_documents(documents)
                    st.toast(f"Loaded {get_collection().count()} KB chunks.", icon="📚")


# -------------------------
# Placeholder Functions for other modules (omitted logic for brevity)
# -------------------------
def patient_input_form(key_prefix="p"):
    # This is a placeholder function for the input form used in other modules
    with st.form(key=f"form_{key_prefix}"):
        st.columns(2)
        submitted = st.form_submit_button("Run Analysis")
    data = {'age': 45, 'gender': 'Male', 'bmi': 25.0, 'sbp': 120.0, 'dbp': 80.0, 'glucose': 100.0, 'cholesterol': 180.0, 'smoker': False}
    return submitted, data

# -------------------------
# Module: Risk Stratification
# -------------------------
if menu == "🧑‍⚕️ Risk Stratification":
    st.title("Risk Stratification")
    st.write("Predict a patient's risk level based on key health indicators.")
    submitted, pdata = patient_input_form("risk")
    if submitted:
        # Simplified risk calculation logic
        score = 0
        score += (pdata['age'] >= 60) * 2
        label = "Low Risk" if score <= 1 else "High Risk"
        st.success(f"Predicted Risk Level: *{label}* (Score: {score})")

# ... (Other module code omitted for brevity) ...
elif menu == "⏱ Length of Stay Prediction":
    st.title("Length of Stay Prediction")
    st.write("Predicts the expected hospital length of stay (in days) for a patient.")
    submitted, pdata = patient_input_form("los")
    if submitted:
        los_est_rounded = 5
        st.success(f"Predicted length of stay: *{los_est_rounded} days*")
elif menu == "👥 Patient Segmentation":
    st.title("Patient Segmentation")
    st.write("Assigns a patient to a distinct health cohort.")
elif menu == "🩻 Imaging Diagnostics":
    st.title("Imaging Diagnostics")
    st.write("Simulates medical image analysis using a dummy model.")
elif menu == "📈 Sequence Forecasting":
    st.title("Sequence Forecasting")
    st.write("Predicts a patient's next health metric value based on a time-series of past data.")
elif menu == "📝 Clinical Notes Analysis":
    st.title("Clinical Notes Analysis")
    st.write("Analyzes clinical notes to provide insights.")
elif menu == "🌐 Translator":
    st.title("Translator")
    st.write("Translate clinical or patient-facing text between different languages.")
elif menu == "💬 Sentiment Analysis":
    st.title("Patient Feedback Sentiment Analysis")
    st.write("Analyzes patient feedback to determine the sentiment.")
elif menu == "💡 Gemini Chat Assistant":
    st.title("Gemini AI Chat Assistant")
    st.write("Ask questions and get information from the powerful Gemini model.")
    # Gemini Chat Assistant logic (as before)
    if not GEMINI_API_KEY:
        st.error("Gemini AI API key is not configured.")
        st.stop()
    if "messages_gemini" not in st.session_state:
        st.session_state["messages_gemini"] = [{"role": "assistant", "content": "Hello! I am a general health assistant powered by Gemini."}]
    
    for msg in st.session_state.messages_gemini:
        st.chat_message(msg["role"]).write(msg["content"])
    
    if prompt := st.chat_input("Ask me anything about general health..."):
        st.session_state.messages_gemini.append({"role": "user", "content": prompt})
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
                st.session_state.messages_gemini.append({"role": "assistant", "content": full_response})


# -------------------------
# Module: RAG Chatbot (The Main Focus)
# -------------------------
elif menu == "🧠 RAG Chatbot":
    
    # --- RAG Settings in Sidebar (Language Selection & KB Management) ---
    st.sidebar.markdown("---")
    st.sidebar.header("RAG Settings")

    # Language selection for RAG output
    st.session_state.selected_language = st.sidebar.selectbox(
        "Select Response Language", 
        list(LANGUAGE_DICT.keys()), 
        index=0, 
        key="rag_lang_select_sidebar"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Knowledge Base Management")
    
    # Display current KB status (Checked *after* load attempt)
    kb_count = get_collection().count()
    st.sidebar.info(f"Knowledge Base Chunks: **{kb_count}**")
    
    # Button to reset and reload KB
    if st.sidebar.button("Reset/Reload Default KB", key="reset_kb_button"):
        clear_and_reload_kb()
        st.toast("Knowledge Base reset and reloaded with default health data!", icon="🔄")


    # --- Main Chat Interface ---
    
    st.markdown("## Health RAG Chatbot 🧠")
    st.markdown("A specialized **Health Consultant AI**. It uses its fixed knowledge base (KB) first, then falls back to **Google Search** to provide comprehensive answers and **reliable external source links** for all health queries.")

    # Initialize RAG chat history
    if "messages_rag" not in st.session_state:
        st.session_state["messages_rag"] = [
            {"role": "assistant", "content": f"Hello! I'm your RAG medical assistant. I have a large health knowledge base, and I will **search the web for external sources** if needed. How can I help clarify your health doubts today?"}
        ]

    # Display chat messages
    for msg in st.session_state.messages_rag:
        st.chat_message(msg["role"]).write(msg["content"])

    # Handle user input
    if prompt := st.chat_input("Ask a health question..."):
        if not st.session_state.get('gemini_client'):
            st.chat_message("assistant").write("The RAG chatbot is not configured due to missing Gemini API key or dependencies.")
            st.stop()
        
        st.session_state.messages_rag.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating comprehensive response..."):
                # Call the RAG pipeline with the LLM/Google Search fallback logic
                full_response = rag_pipeline(st.session_state.messages_rag[-1]["content"], st.session_state.selected_language)
                
                st.write(full_response)
                
                st.session_state.messages_rag.append({"role": "assistant", "content": full_response})
