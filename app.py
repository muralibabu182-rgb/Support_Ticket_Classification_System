import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import string
import io
from nltk.corpus import stopwords
import nltk
import altair as alt
import os

# ----------------- Configuration -----------------
st.set_page_config(
    page_title="Ticket Classifier",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Custom CSS --------------------
st.markdown("""
<style>
/* Font overrides */
* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important; }

/* Custom top banner */
.top-banner {
    background: linear-gradient(135deg, #1E1B4B 0%, #2D2A4A 100%);
    border: 1px solid #4338CA;
    padding: 28px 32px;
    border-radius: 12px;
    margin-bottom: 25px;
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
}
.banner-title {
    color: #A78BFA;
    font-size: 34px;
    font-weight: 800;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 15px;
    letter-spacing: -0.5px;
}
.banner-subtitle {
    color: #9CA3AF;
    font-size: 14px;
    margin-top: 6px;
    margin-bottom: 0;
}

/* Sidebar Custom Stat Cards */
.sidebar-stat-card {
    background-color: #262738;
    border: 1px solid #3B3C54;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
    margin-bottom: 12px;
}
.sidebar-stat-value {
    color: #A78BFA;
    font-size: 24px;
    font-weight: 700;
    margin: 0;
}
.sidebar-stat-label {
    color: #9CA3AF;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 2px;
}

/* Classification Result Card */
.result-card {
    background: linear-gradient(135deg, #262738 0%, #32334A 100%);
    border: 1px solid #4F46E5;
    border-radius: 12px;
    padding: 26px;
    text-align: center;
    margin-top: 10px;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.15);
}
.result-label {
    color: #A78BFA;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 10px;
    font-weight: 600;
}
.result-value {
    color: #FFFFFF;
    font-size: 26px;
    font-weight: 800;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
}
.confidence-badge {
    display: inline-block;
    background-color: rgba(167, 139, 250, 0.15);
    border: 1px solid #7C3AED;
    color: #DDD6FE;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    margin-top: 12px;
}

/* Small lists formatting in sidebar */
.sidebar-list {
    list-style-type: none;
    padding: 0;
    font-size: 13px;
    color: #E2E8F0;
}
.sidebar-list li {
    padding: 5px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Text area styling */
div[data-baseweb="textarea"] > div {
    background-color: #20212E !important;
    border-color: #3B3C54 !important;
    color: #F3F4F6 !important;
    border-radius: 8px !important;
}

/* Primary Button Styling */
button[kind="primary"] {
    background: linear-gradient(90deg, #6366F1 0%, #8B5CF6 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
}

/* Secondary Buttons */
button[kind="secondary"] {
    background-color: #262738 !important;
    border: 1px solid #3B3C54 !important;
    color: #E2E8F0 !important;
    border-radius: 6px !important;
}
button[kind="secondary"]:hover {
    border-color: #8B5CF6 !important;
    color: #A78BFA !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------- Helper Functions -----------------
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

@st.cache_resource(show_spinner=False)
def download_stopwords():
    try:
        nltk.download('stopwords', quiet=True)
        return set(stopwords.words('english'))
    except Exception:
        return {'the', 'and', 'is', 'in', 'to', 'of', 'it', 'for', 'a', 'an', 'on', 'at', 'with'}

stop_words = download_stopwords()

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        if os.path.exists("ticket_model.pkl"):
            return joblib.load("ticket_model.pkl")
    except Exception as e:
        st.error(f"Error loading model: {e}")
    return None

model = load_model()

@st.cache_data(show_spinner=False)
def load_data():
    try:
        if os.path.exists("all_tickets_processed_improved_v3.csv"):
            return pd.read_csv("all_tickets_processed_improved_v3.csv")
    except Exception:
        pass
    return None

df = load_data()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

# Category icon mapping
CATEGORY_ICONS = {
    "Hardware": "💻",
    "Access": "🔑",
    "HR Support": "👥",
    "Administrative rights": "🛡️",
    "Internal Project": "📋",
    "Miscellaneous": "📦",
    "Purchase": "🛒",
    "Storage": "💾"
}

# Initialize session state for example texts
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

def set_example(text):
    st.session_state.input_text = text

# ----------------- Sidebar -----------------------
st.sidebar.markdown("### 🎫 Ticket Classifier")
st.sidebar.markdown("<br>", unsafe_allow_html=True)

st.sidebar.markdown("#### 📈 Dataset Stats")

total_tickets = len(df) if df is not None else 3806
unique_cats = df['Topic_group'].nunique() if df is not None else 8

st.sidebar.markdown(f"""
<div class="sidebar-stat-card">
    <p class="sidebar-stat-value">{total_tickets:,}</p>
    <p class="sidebar-stat-label">TOTAL TICKETS</p>
</div>
<div class="sidebar-stat-card">
    <p class="sidebar-stat-value">{unique_cats}</p>
    <p class="sidebar-stat-label">CATEGORIES</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("#### 🏷️ Categories")
st.sidebar.markdown("""
<ul class="sidebar-list">
    <li>🔑 Access</li>
    <li>🛡️ Administrative rights</li>
    <li>👥 HR Support</li>
    <li>💻 Hardware</li>
    <li>📋 Internal Project</li>
    <li>📦 Miscellaneous</li>
    <li>🛒 Purchase</li>
    <li>💾 Storage</li>
</ul>
""", unsafe_allow_html=True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("#### 🤖 Model Specs")
st.sidebar.markdown("""
<ul class="sidebar-list">
    <li>🔹 Vectorizer: TF-IDF (1-2 ngrams)</li>
    <li>🔹 Classifier: LinearSVC</li>
    <li>🔹 Features: 20,000 top n-grams</li>
    <li>🔹 Architecture: Scikit-learn Pipeline</li>
</ul>
""", unsafe_allow_html=True)

# ----------------- Main Body ---------------------
# Top Banner
st.markdown("""
<div class="top-banner">
    <h1 class="banner-title">🎫 Support Ticket Classification System</h1>
    <p class="banner-subtitle">AI & NLP-driven classification • Automatically triage, route, and manage enterprise support tickets</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Single Ticket Classification", 
    "📁 Batch CSV Processing", 
    "📊 Dataset Analytics", 
    "ℹ️ Architecture & About"
])

# ----------------- TAB 1: Single Classification -----------------
with tab1:
    col1, col2 = st.columns([1.2, 1], gap="large")
    
    with col1:
        st.markdown("### ✏️ Enter Ticket Description")
        
        user_input = st.text_area(
            label="Ticket text", 
            value=st.session_state.input_text,
            height=180, 
            label_visibility="collapsed",
            placeholder="E.g., My laptop screen went completely blank and the external monitor is not detected..."
        )
        
        st.markdown("💡 **Try an example prompt:**")
        
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            if st.button("💻 Hardware Issue", use_container_width=True):
                set_example("My laptop screen is completely black after the latest update. The monitor does not turn on.")
                st.rerun()
            if st.button("💾 Storage Request", use_container_width=True):
                set_example("I need more storage space on my network share drive to save heavy project assets.")
                st.rerun()
        with btn_col2:
            if st.button("🔑 Access Problem", use_container_width=True):
                set_example("I cannot login to my VPN account and my active directory password has expired.")
                st.rerun()
            if st.button("🛒 Purchase Order", use_container_width=True):
                set_example("I need approval to purchase a new ergonomic keyboard and noise-cancelling headset.")
                st.rerun()
        with btn_col3:
            if st.button("👥 HR Support", use_container_width=True):
                set_example("How do I update my direct deposit bank details for upcoming monthly payroll?")
                st.rerun()
            if st.button("🛡️ Admin Rights", use_container_width=True):
                set_example("Please grant local administrator privileges on my workstation to install Docker.")
                st.rerun()
                
        st.markdown("<br>", unsafe_allow_html=True)
        classify_clicked = st.button("🚀 Classify Ticket", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Classification Result")
        
        if classify_clicked:
            if not user_input.strip():
                st.warning("⚠️ Please enter a ticket description before predicting.")
            elif model is None:
                st.error("⚠️ Model file (`ticket_model.pkl`) not found. Please run `python train.py` first.")
            else:
                with st.spinner("Analyzing ticket text..."):
                    cleaned = clean_text(user_input)
                    clf = model.named_steps["clf"]
                    classes = clf.classes_
                    
                    scores = model.decision_function([cleaned])
                    if scores.ndim == 1:
                        scores = scores.reshape(1, -1)
                    probs = softmax(scores)[0]
                    
                    pred_idx = np.argmax(probs)
                    result = classes[pred_idx]
                    confidence = probs[pred_idx] * 100
                    
                    icon = CATEGORY_ICONS.get(result, "🎯")
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <p class="result-label">PREDICTED CATEGORY</p>
                        <div class="result-value">{icon} {result}</div>
                        <div class="confidence-badge">Confidence: {confidence:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Top probabilities breakdown chart
                    prob_df = pd.DataFrame({
                        "Category": classes,
                        "Confidence": probs * 100
                    }).sort_values(by="Confidence", ascending=False)
                    
                    chart = alt.Chart(prob_df).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
                        x=alt.X("Confidence:Q", title="Probability (%)", scale=alt.Scale(domain=[0, 100])),
                        y=alt.Y("Category:N", sort="-x", title=None),
                        color=alt.condition(
                            alt.datum.Category == result,
                            alt.value("#8B5CF6"),  # Highlight predicted
                            alt.value("#4B487A")   # Others
                        ),
                        tooltip=[alt.Tooltip("Category:N"), alt.Tooltip("Confidence:Q", format=".1f")]
                    ).properties(height=230)
                    
                    st.markdown("##### 📊 Category Probabilities")
                    st.altair_chart(chart, use_container_width=True)
                    
                    # Text summary metrics
                    raw_words = len(user_input.split())
                    clean_words = len(cleaned.split())
                    m1, m2 = st.columns(2)
                    m1.metric("Raw Word Count", raw_words)
                    m2.metric("Filtered Keywords", clean_words)
        else:
            # Clean empty state
            st.markdown("""
            <div class="result-card" style="opacity: 0.6;">
                <p class="result-label">PREDICTED CATEGORY</p>
                <div class="result-value">⏳ Awaiting ticket input...</div>
                <p style="color: #9CA3AF; font-size: 13px; margin-top: 15px;">
                    Enter a ticket description or select an example above and click <b>Classify Ticket</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)

# ----------------- TAB 2: Batch CSV Processing -----------------
with tab2:
    st.markdown("### 📁 Batch Ticket Classification")
    st.markdown("Upload a CSV containing multiple tickets to automatically classify and export the results.")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(batch_df):,}** rows.")
            
            # Select column containing ticket text
            text_columns = [col for col in batch_df.columns if batch_df[col].dtype == 'object']
            default_idx = 0
            if "Document" in text_columns:
                default_idx = text_columns.index("Document")
            elif "text" in text_columns:
                default_idx = text_columns.index("text")
            elif "description" in text_columns:
                default_idx = text_columns.index("description")
                
            selected_col = st.selectbox("Select column containing ticket text:", text_columns, index=default_idx)
            
            if st.button("⚡ Process & Classify Batch", type="primary"):
                if model is None:
                    st.error("Model not found. Please train the model first.")
                else:
                    with st.spinner(f"Classifying {len(batch_df)} tickets..."):
                        cleaned_series = batch_df[selected_col].apply(clean_text)
                        clf = model.named_steps["clf"]
                        classes = clf.classes_
                        
                        scores = model.decision_function(cleaned_series)
                        probs = softmax(scores)
                        
                        pred_indices = np.argmax(probs, axis=1)
                        batch_df["Predicted_Category"] = classes[pred_indices]
                        batch_df["Confidence_%"] = np.round(np.max(probs, axis=1) * 100, 1)
                        
                        st.success(f" Successfully classified {len(batch_df):,} tickets!")
                        
                        # Category distribution chart
                        dist_df = batch_df["Predicted_Category"].value_counts().reset_index()
                        dist_df.columns = ["Category", "Count"]
                        
                        dist_chart = alt.Chart(dist_df).mark_bar(cornerRadiusTopRight=4, cornerRadiusTopLeft=4).encode(
                            x=alt.X("Category:N", sort="-y", title="Predicted Category"),
                            y=alt.Y("Count:Q", title="Number of Tickets"),
                            color=alt.Color("Category:N", scale=alt.Scale(scheme="purples"), legend=None),
                            tooltip=["Category", "Count"]
                        ).properties(height=300)
                        
                        st.altair_chart(dist_chart, use_container_width=True)
                        
                        # Preview table
                        st.markdown("#### 📋 Results Preview")
                        st.dataframe(batch_df.head(50), use_container_width=True)
                        
                        # CSV Download button
                        csv_buffer = io.StringIO()
                        batch_df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="⬇️ Download Classified CSV",
                            data=csv_buffer.getvalue(),
                            file_name="classified_tickets_output.csv",
                            mime="text/csv",
                            type="primary"
                        )
        except Exception as ex:
            st.error(f"Error processing CSV: {ex}")
    else:
        st.info("💡 You can test this feature using `all_tickets_processed_improved_v3.csv` from the project directory.")

# ----------------- TAB 3: Dataset Analytics -----------------
with tab3:
    st.markdown("### 📊 Dataset Analytics & Category Distribution")
    if df is not None:
        cat_counts = df['Topic_group'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Count']
        cat_counts['Percentage'] = (cat_counts['Count'] / cat_counts['Count'].sum() * 100).round(1)
        
        c1, c2 = st.columns([1.5, 1])
        with c1:
            main_chart = alt.Chart(cat_counts).mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5).encode(
                x=alt.X('Count:Q', title='Total Tickets'),
                y=alt.Y('Category:N', sort='-x', title=None),
                color=alt.Color('Count:Q', scale=alt.Scale(scheme='purples'), legend=None),
                tooltip=['Category', 'Count', alt.Tooltip('Percentage:Q', format='.1f')]
            ).properties(height=360)
            st.altair_chart(main_chart, use_container_width=True)
        
        with c2:
            st.markdown("#### 📈 Summary Breakdown")
            st.dataframe(cat_counts, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("#### 🔍 Browse Raw Dataset Records")
        selected_category_filter = st.selectbox(
            "Filter by category:", 
            ["All"] + list(df['Topic_group'].unique())
        )
        if selected_category_filter != "All":
            filtered_df = df[df['Topic_group'] == selected_category_filter]
        else:
            filtered_df = df
        st.dataframe(filtered_df.head(100), use_container_width=True)
    else:
        st.warning("Dataset file `all_tickets_processed_improved_v3.csv` was not found.")

# ----------------- TAB 4: About -----------------
with tab4:
    st.markdown("### ℹ️ About the Architecture & Pipeline")
    st.markdown("""
    This application utilizes Natural Language Processing (NLP) and Machine Learning to automate enterprise ticket triage.
    
    #### 🏗️ Architecture Pipeline
    1. **Text Ingestion**: User provides unstructured support ticket descriptions.
    2. **NLP Text Preprocessing**:
       - Lowercasing and digit/punctuation removal.
       - English stopword elimination via NLTK.
       - Removal of short noisy tokens (< 3 chars).
    3. **Feature Extraction**:
       - **TF-IDF Vectorization** using unigrams + bigrams (`ngram_range=(1,2)`).
       - Top 20,000 informative n-gram features.
    4. **Classification Engine**:
       - **Linear Support Vector Classifier (`LinearSVC`)**.
       - Scalable high-dimensional hyperplane optimization.
       - Calibrated multiclass decision boundary probabilities.
    
    #### 📁 Project Scripts
    - `train.py`: Standalone script to train, evaluate, and save `ticket_model.pkl`.
    - `predict.py`: CLI inference utility for interactive, single, or batch CSV prediction.
    - `app.py`: Streamlit dashboard web application.
    """)
