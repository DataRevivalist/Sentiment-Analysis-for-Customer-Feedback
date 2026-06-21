"""
app.py
------
Streamlit application for real-time sentiment prediction on customer reviews.

Run with:
    streamlit run src/app.py

Requirements:
    - Run Notebook 07 first to generate model artefacts in models/
    - Install dependencies: pip install -r requirements.txt
"""

import sys
import os

# Allow imports from the project root regardless of where the app is launched from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

from src.predict import load_artefacts, predict_single, predict_batch

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ShopEase Sentiment Analyser",
    page_icon="shopping_bags",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load model artefacts (cached so they are loaded once per session)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_artefacts():
    models_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
    )
    return load_artefacts(models_dir=models_dir)


try:
    vectoriser, classifier, label_encoder = get_artefacts()
    MODEL_LOADED = True
except FileNotFoundError as e:
    MODEL_LOADED = False
    LOAD_ERROR = str(e)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

SENTIMENT_COLOURS = {
    "Positive": "green",
    "Negative": "red",
    "Neutral":  "blue",
}

SENTIMENT_ICONS = {
    "Positive": "check_mark_button",
    "Negative": "cross_mark",
    "Neutral":  "white_circle",
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("ShopEase Europe — Real-Time Sentiment Analyser")
st.markdown(
    "Classify customer reviews instantly using the trained machine learning pipeline. "
    "Use the **Single Review** tab to analyse one review or the **Batch Upload** tab "
    "to process a CSV file of reviews."
)

if not MODEL_LOADED:
    st.error(
        f"Model artefacts could not be loaded: {LOAD_ERROR}\n\n"
        "Run **Notebook 07** first to train and save the model, then restart this app."
    )
    st.stop()

st.success("Model loaded successfully.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["Single Review", "Batch Upload", "About"])

# ── Tab 1: Single Review ────────────────────────────────────────────────────
with tab1:
    st.subheader("Analyse a Single Review")
    review_input = st.text_area(
        "Paste a customer review below",
        height=140,
        placeholder="e.g. The item arrived broken and customer service refused to help. Very disappointed.",
    )

    if st.button("Analyse Sentiment", type="primary"):
        if review_input.strip():
            result = predict_single(review_input, vectoriser, classifier, label_encoder)
            label      = result["label"]
            confidence = result["confidence"]
            probs      = result["probabilities"]
            colour     = SENTIMENT_COLOURS.get(label, "grey")

            st.markdown(f"### Prediction: :{colour}[**{label}**]")
            st.markdown(f"**Confidence:** {confidence * 100:.1f}%")

            st.markdown("**Probability breakdown:**")
            for cls in ["Negative", "Neutral", "Positive"]:
                prob = probs.get(cls, 0.0)
                st.progress(
                    int(prob * 100),
                    text=f"{cls}: {prob * 100:.1f}%",
                )
        else:
            st.warning("Please enter a review before clicking Analyse.")

# ── Tab 2: Batch Upload ─────────────────────────────────────────────────────
with tab2:
    st.subheader("Batch Prediction from CSV")
    st.markdown(
        "Upload a CSV file with a column named **`review`**. "
        "The app will add a `predicted_sentiment` column and a confidence score "
        "for each row."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
            st.stop()

        if "review" not in batch_df.columns:
            st.error(
                "The uploaded CSV must contain a column named `review`. "
                f"Columns found: {batch_df.columns.tolist()}"
            )
        else:
            st.info(f"Processing {len(batch_df):,} reviews...")
            results = predict_batch(
                batch_df["review"].astype(str).tolist(),
                vectoriser,
                classifier,
                label_encoder,
            )
            batch_df["predicted_sentiment"] = [r["label"] for r in results]
            batch_df["confidence"]          = [round(r["confidence"] * 100, 1) for r in results]

            st.success(f"Done. {len(batch_df):,} reviews classified.")
            st.dataframe(
                batch_df[["review", "predicted_sentiment", "confidence"]].head(100),
                use_container_width=True,
            )

            csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download predictions as CSV",
                data=csv_bytes,
                file_name="shopease_predictions.csv",
                mime="text/csv",
            )

            # Summary chart
            st.markdown("**Prediction summary:**")
            summary = batch_df["predicted_sentiment"].value_counts()
            col1, col2, col3 = st.columns(3)
            col1.metric("Negative", summary.get("Negative", 0))
            col2.metric("Positive", summary.get("Positive", 0))
            col3.metric("Neutral",  summary.get("Neutral",  0))

# ── Tab 3: About ────────────────────────────────────────────────────────────
with tab3:
    st.subheader("About This Application")
    st.markdown("""
**Project:** ShopEase Europe — Sentiment Analysis for Customer Feedback

**Model:** Logistic Regression with TF-IDF features (balanced class weights)

**Training data:** 20,407 Amazon customer reviews (2007 to 2024)

**Classes:** Negative (68%), Positive (28%), Neutral (4%)

**Categories covered:** Sports, Electronics, Fashion, Beauty, Toys, Home and Living, Food and Grocery

**Pipeline:**
1. Text is lowercased and cleaned (URLs, HTML, emojis removed)
2. Domain-specific stop words are removed (including 'amazon', 'order', 'review')
3. Tokens are lemmatised using WordNet
4. The cleaned text is transformed using a TF-IDF vectoriser (20,000 features, unigrams and bigrams)
5. The Logistic Regression classifier predicts a sentiment class and returns probability scores

**Limitations:**
- The model was trained on English-language reviews. Non-English input may produce unreliable results.
- Sarcasm and irony are difficult to classify correctly without contextual understanding.
- The Neutral class has very few training examples (4%) and may be misclassified more often than the other classes.

**Source code:** github.com/your-username/shopease-sentiment-analysis
    """)
