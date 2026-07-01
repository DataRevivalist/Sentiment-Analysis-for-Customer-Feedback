
import streamlit as st
import pandas as pd
import pickle
import glob
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

for pkg in ["punkt", "stopwords", "wordnet", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

# Page Configuration
st.set_page_config(
    page_title="ShopEase Sentiment Analyser",
    page_icon="Shopping Bags",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #F0F4F8; }
    .stApp { background-color: #F0F4F8; }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .negative { color: #E74C3C; font-size: 32px; font-weight: bold; }
    .positive { color: #27AE60; font-size: 32px; font-weight: bold; }
    .neutral  { color: #2980B9; font-size: 32px; font-weight: bold; }
    .header-banner {
        background-color: #0D1B2A;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header-banner">
        <h1 style="color: #22D3EE; margin: 0;">ShopEase Europe</h1>
        <p style="color: #94A3B8; margin: 0;">
            Sentiment Analysis System  |  Real-Time Customer Review Classification
        </p>
    </div>
""", unsafe_allow_html=True)

# Load Model Artefacts
@st.cache_resource
def load_models():
    try:
        with open("models/tfidf_vectoriser.pkl", "rb") as f:
            tfidf = pickle.load(f)
        with open("models/label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        model_path = glob.glob("models/best_model_*.pkl")[0]
        with open(model_path, "rb") as f:
            clf = pickle.load(f)
        return tfidf, le, clf, True
    except Exception as e:
        st.sidebar.error(f"Model load error: {e}")
        return None, None, None, False

tfidf, le, clf, model_loaded = load_models()

# Text Preprocessing
lemmatizer = WordNetLemmatizer()
STOPS = set(stopwords.words("english")) | {
    "product", "item", "ordered", "order", "amazon",
    "purchase", "bought", "buy", "would", "also", "one",
    "get", "got", "use", "used", "using", "review",
    "star", "stars", "rating"
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?\\S+|www\\.\\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t)
              for t in tokens if t not in STOPS and len(t) > 2]
    return " ".join(tokens)

def predict_sentiment(text):
    if not model_loaded:
        return "Unknown", 0.0, {}
    cleaned   = clean_text(text)
    X         = tfidf.transform([cleaned])
    pred_idx  = clf.predict(X)[0]
    pred_prob = clf.predict_proba(X)[0]
    label     = le.inverse_transform([pred_idx])[0]
    probs     = {cls: round(float(p), 4)
                 for cls, p in zip(le.classes_, pred_prob)}
    return label, float(max(pred_prob)), probs

# Sidebar
st.sidebar.markdown("## ShopEase Europe")
st.sidebar.markdown("Sentiment Analysis System")
st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Single Review", "Batch Analysis", "Dashboard Summary", "About"]
)
st.sidebar.markdown("---")
if model_loaded:
    st.sidebar.success("Model loaded successfully")
else:
    st.sidebar.error("Model not found. Run Notebook 7 first.")

# PAGE 1 - Single Review
if page == "Single Review":
    st.subheader("Analyse a Single Customer Review")
    st.markdown(
        "Paste any customer review below to instantly classify "
        "its sentiment and see the confidence breakdown."
    )

    review_input = st.text_area(
        "Customer Review",
        height=150,
        placeholder="e.g. My account was frozen for two weeks and nobody could help me."
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyse_btn = st.button(
            "Analyse Sentiment",
            type="primary",
            use_container_width=True
        )

    if analyse_btn:
        if not review_input.strip():
            st.warning("Please enter a review before clicking Analyse.")
        elif not model_loaded:
            st.error("Model not loaded. Please run Notebook 7 first.")
        else:
            with st.spinner("Analysing..."):
                label, confidence, probs = predict_sentiment(review_input)

            colour_map = {
                "Negative": "negative",
                "Positive": "positive",
                "Neutral":  "neutral"
            }

            st.markdown("---")
            st.markdown("### Result")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <p style="color:#64748B; margin:0;">Prediction</p>
                        <p class="{colour_map[label]}">{label.upper()}</p>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <p style="color:#64748B; margin:0;">Confidence</p>
                        <p style="font-size:32px; font-weight:bold; color:#0891B2;">
                            {confidence*100:.1f}%
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            with col3:
                uncertain = "YES" if confidence < 0.65 else "NO"
                u_colour  = "#E74C3C" if confidence < 0.65 else "#27AE60"
                st.markdown(f"""
                    <div class="metric-card">
                        <p style="color:#64748B; margin:0;">Needs Human Review</p>
                        <p style="font-size:32px; font-weight:bold; color:{u_colour};">
                            {uncertain}
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("#### Probability Breakdown")
            for cls in ["Negative", "Neutral", "Positive"]:
                prob = probs.get(cls, 0.0)
                st.markdown(f"**{cls}**")
                st.progress(float(prob))
                st.caption(f"{prob*100:.1f}%")

            if confidence < 0.65:
                st.warning(
                    "Confidence is below 65%. This review has been "
                    "flagged for human review."
                )

# PAGE 2 - Batch Analysis
elif page == "Batch Analysis":
    st.subheader("Batch Review Analysis")
    st.markdown(
        "Upload a CSV file with a column named **review** to classify "
        "multiple reviews at once and download the results."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

        if "review" not in batch_df.columns:
            st.error(
                f"CSV must have a column named review. "
                f"Columns found: {batch_df.columns.tolist()}"
            )
        elif not model_loaded:
            st.error("Model not loaded. Please run Notebook 7 first.")
        else:
            st.info(f"Processing {len(batch_df):,} reviews...")
            progress_bar = st.progress(0.0)
            results = []

            for i, row in enumerate(batch_df["review"].astype(str)):
                label, conf, probs = predict_sentiment(row)
                results.append({
                    "predicted_sentiment": label,
                    "confidence_pct":      round(conf * 100, 1),
                    "needs_human_review":  conf < 0.65
                })
                progress_bar.progress((i + 1) / len(batch_df))

            result_df = pd.concat(
                [batch_df, pd.DataFrame(results)], axis=1
            )

            st.success(f"Done. {len(result_df):,} reviews classified.")
            st.markdown("---")

            col1, col2, col3, col4 = st.columns(4)
            neg_c = (result_df["predicted_sentiment"] == "Negative").sum()
            pos_c = (result_df["predicted_sentiment"] == "Positive").sum()
            neu_c = (result_df["predicted_sentiment"] == "Neutral").sum()
            unc_c = result_df["needs_human_review"].sum()

            col1.metric("Negative", f"{neg_c:,}",
                        f"{neg_c/len(result_df)*100:.1f}%")
            col2.metric("Positive", f"{pos_c:,}",
                        f"{pos_c/len(result_df)*100:.1f}%")
            col3.metric("Neutral",  f"{neu_c:,}",
                        f"{neu_c/len(result_df)*100:.1f}%")
            col4.metric("Flagged for Review", f"{unc_c:,}",
                        f"{unc_c/len(result_df)*100:.1f}%")

            st.markdown("#### Preview (first 50 rows)")
            st.dataframe(
                result_df[[
                    "review", "predicted_sentiment",
                    "confidence_pct", "needs_human_review"
                ]].head(50),
                use_container_width=True
            )

            csv_out = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Full Predictions CSV",
                data=csv_out,
                file_name="shopease_predictions.csv",
                mime="text/csv"
            )

# PAGE 3 - Dashboard Summary
elif page == "Dashboard Summary":
    st.subheader("Project Dashboard Summary")
    st.markdown("Key findings from the ShopEase Europe sentiment analysis project.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews",   "20,407")
    col2.metric("Negative Rate",   "68.2%",
                delta="-8.2% target", delta_color="inverse")
    col3.metric("Avg Star Rating", "2.19",  delta="+0.61 target")
    col4.metric("Best Model F1",   "0.89",  delta="DistilBERT")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Sentiment Distribution")
        sent_data = pd.DataFrame({
            "Sentiment":  ["Negative", "Positive", "Neutral"],
            "Count":      [14157, 5424, 825],
            "Percentage": ["68.2%", "27.6%", "4.2%"]
        })
        st.dataframe(sent_data, use_container_width=True, hide_index=True)

        st.markdown("### Negative Rate by Category")
        cat_data = pd.DataFrame({
            "Category": ["Home and Living", "Toys", "Food and Grocery",
                         "Electronics", "Sports", "Beauty", "Fashion"],
            "Negative Rate": ["70.0%","69.3%","68.4%",
                              "68.2%","67.7%","67.5%","66.0%"]
        })
        st.dataframe(cat_data, use_container_width=True, hide_index=True)

    with col_right:
        st.markdown("### Model Performance Ranking")
        model_data = pd.DataFrame({
            "Rank":     ["1st","2nd","3rd","4th","5th","6th"],
            "Model":    ["DistilBERT","Logistic Regression","XGBoost",
                         "Random Forest","Bi-LSTM","Complement Naive Bayes"],
            "F1 Score": [0.89, 0.87, 0.85, 0.84, 0.83, 0.79],
            "ROC-AUC":  [0.95, 0.93, 0.92, 0.91, 0.90, 0.88]
        })
        st.dataframe(model_data, use_container_width=True, hide_index=True)

        st.markdown("### Top Complaint Themes")
        topic_data = pd.DataFrame({
            "Topic": ["Account and Billing","Returns and Refunds",
                      "Customer Service","Delivery and Shipping",
                      "Product Quality","Pricing and Value"],
            "Negative Rate": ["92%","90%","88%","85%","78%","72%"]
        })
        st.dataframe(topic_data, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Business Recommendations")
    recs = [
        ("CRITICAL", "Fix account management — 92% negative rate in this topic"),
        ("HIGH",     "Accelerate returns and refunds — 90% negative rate"),
        ("HIGH",     "Overhaul customer service — 5,998 complaint mentions"),
        ("MEDIUM",   "Strengthen delivery reliability — highest complaint volume"),
        ("MEDIUM",   "Deploy automated sentiment pipeline to replace manual review"),
    ]
    priority_colours = {
        "CRITICAL": "#E74C3C",
        "HIGH":     "#F39C12",
        "MEDIUM":   "#2980B9"
    }
    for priority, rec in recs:
        colour = priority_colours[priority]
        st.markdown(
            f'<span style="background:{colour}; color:white; '
            f'padding:2px 8px; border-radius:4px; font-size:11px; '
            f'font-weight:bold;">{priority}</span> {rec}',
            unsafe_allow_html=True
        )

# PAGE 4 - About
elif page == "About":
    st.subheader("About This Application")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Project Details")
        st.markdown("""
        **Project:** ShopEase Europe Sentiment Analysis

        **Dataset:** 20,407 Amazon customer reviews

        **Date Range:** 2007 to 2024

        **Countries:** 148 unique countries

        **Categories:** Sports, Electronics, Fashion, Beauty,
        Toys, Home and Living, Food and Grocery
        """)

        st.markdown("### Model Details")
        st.markdown("""
        **Primary Model:** Logistic Regression with TF-IDF

        **Advanced Model:** DistilBERT (fine-tuned)

        **Best F1 Score:** 0.89 (DistilBERT)

        **Imbalance Handling:** Balanced class weights

        **Uncertainty Threshold:** 65% confidence
        """)

    with col2:
        st.markdown("### Pipeline Steps")
        steps = [
            "Text lowercased and cleaned",
            "URLs, HTML, and emojis removed",
            "Domain-specific stop words removed",
            "Tokens lemmatised using WordNet",
            "TF-IDF vectorisation (20,000 features)",
            "Logistic Regression classification",
            "Probability scores returned for all 3 classes",
            "Reviews below 65% confidence flagged for human review"
        ]
        for i, step in enumerate(steps, 1):
            st.markdown(f"**{i}.** {step}")

        st.markdown("### Deployment Stack")
        st.markdown("""
        - **API:** FastAPI for production inference
        - **Dashboard:** Streamlit (this app)
        - **Container:** Docker
        - **Monitoring:** Evidently AI
        - **Retraining:** Apache Airflow quarterly schedule
        - **Model Registry:** MLflow
        """)
