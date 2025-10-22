# app.py
# HealthAI Suite - Intelligent Analytics for Patient Care

import streamlit as st
import os
import sys
import tempfile
import uuid
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# CRITICAL FIX: Ensure Pysqlite3 is used for Streamlit/ChromaDB compatibility
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules['pysqlite3']
except ImportError:
    pass

# Now import RAG libraries
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import necessary placeholders for other modules
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    MarianMTModel,
    MarianTokenizer,
)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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


# -------------------------
# App config
# -------------------------
st.set_page_config(page_title="HealthAI Suite", page_icon="🩺", layout="wide")

# Gemini AI config
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
A migraine is a neurological condition causing severe, throbbing headaches, often with nausea and light/sound sensitivity. Triggers include stress, certain foods (aged cheese, wine), and sleep changes. Acute treatment uses triptans or NSAIDs; prevention involves daily medications like beta-blockers.

### Asthma Control
Asthma is a chronic condition where airways narrow and swell. Symptoms include wheezing, shortness of breath, and coughing. Treatment relies on two main types of inhalers: **Reliever** and **Preventer**. A written Asthma Action Plan is crucial.

### Mental Health: Depression and Anxiety
**Depression** is a persistent feeling of sadness and loss of interest. Treatment includes psychotherapy (e.g., CBT) and antidepressant medications. **Anxiety disorders** involve excessive worry and fear; treatment also includes therapy, medication, and mindfulness techniques.

### Nutrition and Diet
A balanced diet is essential for cardiovascular health. Key components include: high intake of fruits, vegetables, and whole grains; lean proteins; healthy fats; and limited consumption of processed foods and added sugars. **The Mediterranean Diet** is widely recommended.

### Exercise Guidelines
Adults should aim for at least **150 minutes of moderate-intensity** aerobic exercise per week. Strength training should be done for all major muscle groups at least two days per week.

### Basic First Aid and Heart Health
For minor cuts, clean the area and cover with a sterile bandage. For minor burns, cool immediately with cold running water. Key indicators of a healthy heart include normal blood pressure and healthy cholesterol levels.
"""

# -------------------------
# Helpers: RAG/Gemini Initialization
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
        pass # Ignore if collection doesn't exist
    
    # 2. Process and store the default documents
    with st.spinner("Processing new knowledge base..."):
        documents = split_documents(KNOWLEDGE_BASE_TEXT)
        process_and_store_documents(documents)
    
    # Reset chat history
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

def rag_pipeline(query, selected_language):
    """
    Executes the RAG pipeline with an LLM fallback to Google Search tool for OOKB queries.
    """
    collection = get_collection()
    relevant_docs = retrieve_documents(query)
    
    # Determine if we need to use the KB or fall back to external search
    # Fallback if few documents are retrieved (or KB is empty)
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
        
        fallback_query = f"Provide a detailed, consultant-style answer to the user's health question: {query}. Ensure all facts are supported by your search results."
        
        response_json = call_gemini_api(
            prompt=fallback_query, 
            system_instruction=system_instruction,
            tools=st.session_state.google_search_tool # Use the search tool
        )
        
    else:
        # --- Internal RAG (KB Context) ---
        
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
        return f"An error occurred while generating the response: {response_json['error']}."
    
    try:
        return response_json['response']
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
        st.session_state.db_client, st.session_state.model, st.session_state.gemini_client, st.session_state.google_search_tool = initialize_rag_dependencies()
        
        # Load the default knowledge base only if it's the RAG Chatbot AND the KB is empty
        if menu == "🧠 RAG Chatbot":
            if st.session_state.db_client and get_collection().count() == 0:
                with st.spinner("Loading and processing default knowledge base (large)..."):
                    documents = split_documents(KNOWLEDGE_BASE_TEXT)
                    process_and_store_documents(documents)
                    st.toast(f"Loaded {get_collection().count()} KB chunks.", icon="📚")


# -------------------------
# Placeholder Functions for other modules (must be defined for the code to run)
# -------------------------
def patient_input_form(key_prefix="p"):
    # Simplified form for the demo placeholders
    with st.form(key=f"form_{key_prefix}"):
        st.number_input("Age", min_value=0, max_value=120, value=45, key=f"{key_prefix}_age")
        st.number_input("Glucose (mg/dL)", min_value=40, max_value=400, value=100, key=f"{key_prefix}_glucose")
        submitted = st.form_submit_button("Run Analysis")
    data = {'age': 45, 'gender': 'Male', 'bmi': 25.0, 'sbp': 120.0, 'dbp': 80.0, 'glucose': 100.0, 'cholesterol': 180.0, 'smoker': False}
    return submitted, data

# --- Module Placeholders (simplified to fit standard structure) ---
if menu == "🧑‍⚕️ Risk Stratification":
    st.title("Risk Stratification")
    st.write("Predict a patient's risk level based on key health indicators.")
    submitted, pdata = patient_input_form("risk")
    if submitted:
        score = 0
        score += (pdata['age'] >= 60) * 2
        label = "Low Risk" if score <= 1 else "High Risk"
        st.success(f"Predicted Risk Level: *{label}* (Score: {score})")
# ... (rest of the placeholder module logic should follow here) ...


# -------------------------
# Module: RAG Chatbot (The Main Focus)
# -------------------------
if menu == "🧠 RAG Chatbot":
    
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
