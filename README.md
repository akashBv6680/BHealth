# HealthAI Suite: Intelligent Analytics for Patient Care 🩺

## 1. Project Overview

The **HealthAI Suite** is a modular web application built with **Streamlit** that showcases various AI-driven solutions for healthcare data. It demonstrates how machine learning, deep learning, and natural language processing can be applied to patient data to extract valuable insights and support clinical decision-making. The application is organized into distinct, user-friendly modules, each addressing a specific analytical task.

## 2. Technology Stack

* **Framework**: Streamlit
* **Data Handling**: `pandas`, `numpy`
* **Machine Learning**: `scikit-learn` (for pipelines), `mlxtend` (for association rules)
* **Deep Learning**: `PyTorch`, `Hugging Face Transformers` (for NLP models)
* **API Integration**: `together` (for chatbot functionality)
* **Visualization**: `matplotlib`
* **Version Control**: Git, GitHub

---

## 3. The Project Process: From Data to Deployment

This project follows a structured data science and development workflow.

### **Step 1: Data Acquisition and Preparation**
* **Source**: The project uses a **synthetic healthcare dataset** that mimics real-world patient records. This dataset includes a mix of structured (numerical vitals, demographic data) and unstructured (clinical notes) information.
* **Cleaning**: The raw data is cleaned by handling missing values in key columns like `diagnosis`, `medication`, and `procedure` by replacing them with a default value (`'None'`). This ensures the data is suitable for model training.

### **Step 2: Model Selection and Training**
A range of models, from classic ML to advanced DL, were selected based on the specific analytical task.

* **Tabular Analytics (ML)**:
    * **Task**: Predicting a patient's **risk level** and **length of stay**.
    * **Approach**: Simple rule-based logic and placeholder `scikit-learn` pipelines using `RandomForest` models were created to demonstrate the core concepts without requiring complex training on the fly.
* **Natural Language Processing (NLP)**:
    * **Task**: Analyzing **clinical notes** and **patient feedback**.
    * **Approach**: Pre-trained transformer models from the **Hugging Face Hub** were used. These models, such as `cardiffnlp/twitter-roberta-base-sentiment-latest`, are powerful and provide high accuracy for sentiment analysis and emotion classification out-of-the-box, eliminating the need for extensive training.
* **Association Rule Mining**:
    * **Task**: Discovering **relationships** between medical events.
    * **Approach**: The `mlxtend` library was used to implement the **Apriori algorithm**. This algorithm efficiently finds frequent itemsets and generates association rules from transactional data (e.g., co-occurrence of symptoms and diagnoses).
* **Deep Learning (Placeholder)**:
    * **Task**: **Imaging diagnostics** and **sequence forecasting**.
    * **Approach**: Placeholder modules with dummy logic were included. A real implementation would involve training a **Convolutional Neural Network (CNN)** for image analysis and an **RNN/LSTM** for time-series predictions. These models were not trained in the final app due to the lack of a suitable dataset and to keep the app lightweight.

### **Step 3: App Development and Integration**
* **Interface**: The entire application was built using **Streamlit**, which allows for rapid development of interactive, data-driven web apps using only Python. The UI is structured with a sidebar for easy navigation.
* **Functionality**: Each model and analytical process was encapsulated within a dedicated module in `app.py`. This modular design ensures that each component is self-contained and easy to maintain.
* **Caching**: To optimize performance and reduce loading times, Streamlit's `@st.cache_resource` decorator was used to cache heavy models (like the NLP and translation models) in memory, so they are only loaded once.

### **Step 4: Deployment**
* The project is designed to be easily deployed on platforms like **Streamlit Community Cloud** or **Hugging Face Spaces**. Deployment requires a `requirements.txt` file (listing all dependencies) and a `.streamlit/secrets.toml` file to securely store API keys for the chatbot and Hugging Face models.

---

## 4. How to Run Locally

### **Prerequisites**
* Python 3.9+

### **Installation**
1.  **Clone the repository**:
    `git clone https://github.com/your-username/your-repo-name.git`
2.  **Navigate to the project directory**:
    `cd your-repo-name`
3.  **Install dependencies**:
    `pip install -r requirements.txt`
4.  **Create `.streamlit/secrets.toml`**:
    `mkdir .streamlit && cd .streamlit`
    `touch secrets.toml`
    Then, add your API keys:
    ```toml
    HF_ACCESS_TOKEN = "your_hugging_face_token"
    TOGETHER_API_KEY = "your_together_api_key"
    ```
5.  **Run the app**:
    `streamlit run app.py`

---

## 5. License

This project is licensed under the MIT License.
