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


# -------------------------
# App config
# -------------------------
st.set_page_config(page_title="HealthAI Suite", page_icon="🩺", layout="wide")

# Hugging Face login
try:
    if "HF_ACCESS_TOKEN" in st.secrets:
        hf_login(token=st.secrets["HF_ACCESS_TOKEN"], add_to_git_credential=False)
except Exception:
    pass

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
        model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        
        # Initialize Gemini Client and configure Google Search Tool
        if GEMINI_API_KEY and genai:
            gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            # Correct way to define the search tool for the API config
            google_search_tool = [types.Tool(google_search={})] 
        else:
            gemini_client = None
            google_search_tool = None
            # st.error("Gemini AI client not initialized. Check API key and dependencies.")

        return db_client, model, gemini_client, google_search_tool
    except Exception as e:
        st.error(f"An error occurred during RAG dependency initialization: {e}.")
        st.stop()


# The rest of the load functions remain unchanged (omitted for brevity)
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
    
# ... (Other placeholder functions for other modules) ...


# -------------------------
# RAG/Gemini functions
# -------------------------
def get_collection():
    """Retrieves or creates the ChromaDB collection, ensuring dependencies are initialized."""
    if 'db_client' not in st.session_state:
         st.session_state.db_client, st.session_state.model, st.session_state.gemini_client, st.session_state.google_search_tool = initialize_rag_dependencies()
    return st.session_state.db_client.get_or_create_collection(
        name=COLLECTION_NAME
    )

def clear_and_reload_kb():
    """Clears the existing collection and reloads the default KB."""
    db_client = st.session_state.db_client
    
    # 1. Delete the existing collection
    db_client.delete_collection(name=COLLECTION_NAME)
    
    # 2. Get the new, empty collection (and ensure it's in session state)
    collection = get_collection()
    
    # 3. Process and store the default documents
    documents = split_documents(KNOWLEDGE_BASE_TEXT)
    process_and_store_documents(documents)
    
    # Reset chat history to reflect the KB change
    st.session_state["messages_rag"] = [
        {"role": "assistant", "content": f"Hello! I'm your RAG medical assistant. The Knowledge Base has been reset and now contains {collection.count()} chunks. Ask me about common health topics!"}
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
                return {"error": "401 Unauthorized"}
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
        
        # 1. Fallback to Google Search Tool (External Knowledge)
        system_instruction = (
            f"You are a friendly, helpful, and highly knowledgeable health consultant. "
            f"You must use the Google Search tool to find reliable, current health information "
            f"to answer the user's query: '{query}'. "
            f"Your response must be comprehensive, easy to understand, and answer all parts of the user's question, providing advice as a good consultant would. "
            f"You MUST cite all external sources used with numbered links at the end of your response. "
            f"The final response MUST be in {selected_language}."
        )
        
        # The prompt will trigger the Google Search tool use in the model.
        fallback_query = f"Provide a detailed, consultant-style answer to the user's health question: {query}. Ensure all facts are supported by your search results."
        
        response_json = call_gemini_api(
            prompt=fallback_query, 
            system_instruction=system_instruction,
            tools=st.session_state.google_search_tool # Use the configured search tool
        )
        
    else:
        # --- RAG Success Step (Use KB context) ---
        
        context = "\n".join(relevant_docs)
        
        # The RAG prompt instructs Gemini to use the context ONLY.
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
        return "An error occurred while generating the response. Please try again."
    
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

# Initialization block for RAG/Gemini dependencies
if menu == "🧠 RAG Chatbot" or menu == "💡 Gemini Chat Assistant":
    if 'db_client' not in st.session_state:
        # Load all RAG/Gemini dependencies
        st.session_state.db_client, st.session_state.model, st.session_state.gemini_client, st.session_state.google_search_tool = initialize_rag_dependencies()
        
        # Load the default knowledge base if it's the RAG Chatbot AND the KB is empty
        if menu == "🧠 RAG Chatbot":
            if st.session_state.db_client and get_collection().count() == 0:
                with st.spinner("Loading and processing default knowledge base..."):
                    documents = split_documents(KNOWLEDGE_BASE_TEXT)
                    process_and_store_documents(documents)
                    st.toast(f"Loaded {get_collection().count()} KB chunks.", icon="📚")


# ... (Other module code omitted for brevity as they remain unchanged) ...

# -------------------------
# Module: RAG Chatbot
# -------------------------
if menu == "🧠 RAG Chatbot":
    
    # --- RAG Settings in Sidebar ---
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
    st.sidebar.info(f"Knowledge Base Chunks: **{kb_count}** (Expanded Health KB Loaded)")
    
    # Button to reset and reload KB
    if st.sidebar.button("Reset Knowledge Base", key="reset_kb_button"):
        clear_and_reload_kb()
        st.toast("Knowledge Base reset and reloaded with default data!", icon="🔄")


    # --- Main Chat Interface ---
    
    st.markdown("## Health RAG Chatbot 🧠")
    st.markdown("A specialized **Health Consultant AI** that uses its fixed knowledge base and is augmented with **Google Search** to provide comprehensive, sourced answers to all health queries.")

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
