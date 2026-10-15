# 🎫 Support Ticket Classification System

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

An AI/NLP-powered Machine Learning system to automatically classify and route customer support tickets into predefined departments using Natural Language Processing.

---

## 📌 Problem Statement

Enterprise support teams receive thousands of unstructured tickets daily. Manual ticket routing:
- ⏳ Consumes significant operational time
- 🐢 Delays critical issue resolution
- 📉 Decreases customer satisfaction
- ❌ Is prone to human routing errors

This system automates ticket triage by classifying raw customer inquiries into appropriate operational categories with high accuracy.

---

## 📊 Dataset & Categories

The dataset contains **3,806 labeled support tickets** across **8 operational categories**:

| Icon | Category | Example Query |
| :---: | :--- | :--- |
| 💻 | **Hardware** | *"My laptop screen is completely black after the latest update."* |
| 🔑 | **Access** | *"I cannot login to VPN and my password expired."* |
| 👥 | **HR Support** | *"How do I update my direct deposit information for payroll?"* |
| 🛡️ | **Administrative rights** | *"Please grant local admin privileges on my machine to install Docker."* |
| 📋 | **Internal Project** | *"Sprint planning meeting notes and task assignment updates."* |
| 📦 | **Miscellaneous** | *"Mail verification warning attached in external memo."* |
| 🛒 | **Purchase** | *"Requesting purchase order for ergonomic mouse and keyboard."* |
| 💾 | **Storage** | *"Need additional quota on network share drive for project files."* |

---

## 🛠️ Architecture & ML Pipeline

```mermaid
graph LR
    A[Raw Ticket Text] --> B[Text Cleaning & Stopwords]
    B --> C[TF-IDF Vectorizer <br/> Unigrams + Bigrams]
    C --> D[LinearSVC Classifier]
    D --> E[Predicted Category + Confidence]
```

1. **Preprocessing**: Lowercasing, punctuation & digit removal, NLTK stopword filtering, short token removal.
2. **Feature Engineering**: TF-IDF Vectorization with unigrams and bigrams (`max_features=20,000`).
3. **Model**: `LinearSVC` wrapped in a reusable scikit-learn `Pipeline`.
4. **Evaluation**: ~81% overall accuracy with balanced precision and recall.

---

## 📁 Project Structure

```
Support_Ticket_Classification_System/
├── all_tickets_processed_improved_v3.csv   # Labeled dataset (3,806 tickets)
├── train.py                                # Model training & evaluation pipeline
├── predict.py                              # Command-line inference utility
├── app.py                                  # Streamlit web application
├── ticket_model.pkl                        # Serialized ML pipeline model
├── requirements.txt                        # Project dependencies
├── run_app.bat                             # 1-click Windows launcher
└── README.md                               # Project documentation
```

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

Train a fresh model from scratch and view the classification report:

```bash
python train.py
```

### 3. Run Predictions via CLI

**Single Ticket Prediction:**
```bash
python predict.py --text "My monitor is completely blank and won't turn on"
```

**Interactive Mode:**
```bash
python predict.py
```

**Batch CSV Processing:**
```bash
python predict.py --input tickets_input.csv --output tickets_classified.csv --column Document
```

### 4. Launch the Streamlit Web Application

```bash
streamlit run app.py
```
*(Or double-click `run_app.bat` on Windows)*

---

## 🌐 Web App Features

- 🔍 **Live Ticket Classification**: Real-time classification with probability distribution chart and confidence metric.
- 💡 **Interactive Prompts**: Quick 1-click test examples for each ticket category.
- 📁 **Batch CSV Classification**: Upload bulk tickets in CSV format, visualize distribution, and download enriched data.
- 📊 **Dataset Analytics**: Interactive Altair charts and data explorer.
- ℹ️ **Pipeline & Architecture Documentation**: Detailed explanations of model parameters and design.
