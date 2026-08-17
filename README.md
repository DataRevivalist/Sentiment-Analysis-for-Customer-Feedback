# ShopEase Europe - Sentiment Analysis for Customer Feedback

> **Automated sentiment classification and insight generation from real Amazon customer reviews.**

---

## Project Overview

ShopEase Europe is a fast-growing e-commerce company with operations across the United Kingdom, Germany, France, and Spain. The company receives large volumes of customer feedback daily and needed an automated solution to classify sentiment, surface recurring complaint themes, and generate actionable business insights at scale.

This project delivers a complete end-to-end machine learning pipeline that:

- Classifies customer reviews as **Positive**, **Neutral**, or **Negative**
- Identifies key discussion themes using topic modelling
- Surfaces the primary drivers of customer satisfaction and dissatisfaction
- Provides an interactive Power BI dashboard for business stakeholders
- Includes a deployment-ready Streamlit application for real-time prediction

---

## ⚠️ Important: Dataset Change - Why We Switched to Amazon Reviews

### The Original Dataset

The project was initially scoped to use a **synthetic ShopEase Europe customer feedback dataset** provided in the project brief. This dataset was designed to simulate feedback from ShopEase's four primary markets: the United Kingdom, France, Germany, and Spain.

### What We Discovered During Analysis

After completing the full data audit in Notebook 1, the following critical issues were identified with the original synthetic dataset:

| Issue | Detail |
|-------|--------|
| **Artificial sentiment distribution** | The synthetic data had a near-perfectly balanced 33/33/33 split across Negative, Positive, and Neutral classes. Real customer feedback data is never this balanced - dissatisfied customers are consistently more motivated to leave reviews than satisfied ones. |
| **No statistical variation** | Review lengths, sentiment scores, and rating distributions were suspiciously uniform across all product categories and countries, indicating the data was algorithmically generated rather than observed from real customer behaviour. |
| **Models performed unrealistically well** | Every model trained on the synthetic data achieved over 95% accuracy and F1 scores above 0.96. While this might appear impressive, it confirmed that the dataset was too clean and structured to represent a genuine NLP challenge. Any model would perform well on data this artificial. |
| **No generalisability** | The vocabulary used in the synthetic reviews was repetitive and formulaic. Models trained on it would fail to generalise to any real customer feedback because the language patterns bore no resemblance to how customers actually write. |
| **Business insights were meaningless** | Topic modelling and EDA on synthetic data produced findings that could not be acted upon or presented to real stakeholders because they did not reflect genuine customer behaviour or genuine operational problems. |

### The Decision to Switch

The decision was made to replace the synthetic dataset with **real Amazon customer review data** for the following reasons:

**1. Validating the models was the priority**

The core question was: do the machine learning models and the pipeline actually work? Using synthetic data with artificial balance and artificial language patterns could not answer this. The only way to confirm the pipeline functions correctly in a real-world context was to test it against data that reflects real customer behaviour.

**2. Real data produces genuine imbalance**

The Amazon dataset confirmed the expected real-world distribution: **68.2% Negative, 27.6% Positive, 4.2% Neutral**. This is consistent with established research on online review platforms where dissatisfied customers are significantly more motivated to leave feedback. The synthetic data's 33/33/33 split was statistically implausible and would have produced a misleading model.

**3. The models still worked, proving the pipeline is sound**

After switching to the Amazon dataset, all six models were retrained. Performance dropped from the artificial 95%+ on synthetic data to realistic levels between 0.79 and 0.89 F1. This is not a failure, it is evidence that the pipeline is working correctly. The models are now genuinely learning to distinguish sentiment in real, noisy, imbalanced customer language rather than fitting to an artificially clean pattern.

**4. The business insights became actionable**

Real Amazon data produced genuine findings: account management issues driving 92% negative rates, customer service mentioned nearly 6,000 times in complaints, and clear geographic patterns showing Canada and Ireland as higher-dissatisfaction markets. These are insights a real management team could act on. The synthetic data produced no such findings.

### Summary: What the Dataset Switch Confirmed

> **The issue was never with the models or the pipeline. The issue was with the original synthetic dataset.**

The fact that all six models perform robustly on real Amazon data with appropriate F1 scores, sensible confusion matrices, and genuine class separation which confirms that the complete end-to-end pipeline including data cleaning, text preprocessing, feature engineering, model training, evaluation, and deployment is functioning correctly and is production-ready.

---

## Dataset

| Property | Value |
|---|---|
| Source | Amazon customer reviews (real data) |
| File | `data/amazon_reviews_cleaned.csv` |
| Total reviews | 21,055 |
| After cleaning | 20,407 |
| Date range | August 2007 to September 2024 |
| Countries | 148 unique countries |
| Product categories | Sports, Electronics, Fashion, Beauty, Toys, Home and Living, Food and Grocery |
| Sentiment distribution | 68.2% Negative, 27.6% Positive, 4.2% Neutral |
| Language | Predominantly English (confirmed via langdetect in Notebook 3) |

The dataset has a strong negative-majority class distribution, which is consistent with real-world Amazon review behaviour where dissatisfied customers are significantly more motivated to leave feedback than satisfied ones.

---

## Repository Structure

```
shopease-sentiment-analysis/
│
├── data/
│   ├── amazon_reviews_cleaned.csv              # Raw input dataset
│   ├── amazon_reviews_cleaned_processed.csv    # After cleaning (Notebook 2 output)
│   ├── reviews_with_lang.csv                   # After language detection (Notebook 3 output)
│   ├── reviews_preprocessed.csv                # After NLP preprocessing (Notebook 4 output)
│   └── reviews_with_topics.csv                 # After topic modelling (Notebook 6 output)
│
├── notebooks/
│   ├── 01_data_audit.ipynb                     # Data structure, quality, and completeness
│   ├── 02_data_cleaning.ipynb                  # Deduplication, null handling, feature derivation
│   ├── 03_language_detection.ipynb             # Automated language identification per review
│   ├── 04_text_preprocessing.ipynb             # NLP pipeline and feature engineering
│   ├── 05_eda.ipynb                            # Exploratory analysis across sentiment, geography, time
│   ├── 06_topic_modelling.ipynb                # LDA topic discovery and labelling
│   ├── 07_classical_modelling.ipynb            # Naive Bayes, Logistic Regression, Random Forest, XGBoost
│   ├── 08_transformer_modelling.ipynb          # DistilBERT fine-tuning and zero-shot classification
│   ├── 09_explainability_bilstm.ipynb          # SHAP, LIME, Bi-LSTM, BERT discussion
│   ├── 10_summarisation_dashboard.ipynb        # Extractive/abstractive summarisation, Plotly dashboard
│   └── 11_deployment_monitoring.ipynb          # Streamlit app, drift detection, monitoring framework
│
├── models/
│   ├── tfidf_vectoriser.pkl                    # Fitted TF-IDF vectoriser (Notebook 7 output)
│   ├── label_encoder.pkl                       # Label encoder for sentiment classes
│   ├── best_model_Logistic_Regression.pkl      # Best classical model (Notebook 7 output)
│   └── distilbert/                             # Fine-tuned DistilBERT model files
│       ├── config.json
│       ├── pytorch_model.bin
│       └── tokenizer files
│
├── reports/
│   ├── figures/                                # All charts and visualisations saved from notebooks
│   └── ShopEase_Methodology_Justification.docx
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                        # Reusable text cleaning functions
│   ├── predict.py                              # Inference functions for loading and running models
│   └── app.py                                  # Streamlit deployment application
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py                   # Unit tests for text cleaning pipeline
│   └── test_predict.py                         # Unit tests for inference functions
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Five-Day Project Plan

| Day | Notebooks | Tasks |
|-----|-----------|-------|
| Day 1 | 01, 02, 03, 04 | Data audit, cleaning, language detection, text preprocessing and feature engineering |
| Day 2 | 05 | Exploratory data analysis: sentiment, geography, time trends, word clouds, n-grams |
| Day 3 | 06 | Sentiment driver discovery: topic modelling and complaint theme identification |
| Day 4 | 07, 09 | Classical NLP modelling, SHAP, LIME, Bi-LSTM explainability |
| Day 5 | 08, 10, 11 | Transformer model, summarisation dashboard, deployment and monitoring |

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/DataRevivalist/Sentiment-Analysis-for-Customer-Feedback.git
cd Sentiment-Analysis-for-Customer-Feedback
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up the data directory

Place `amazon_reviews_cleaned.csv` inside the `data/` folder. The notebooks reference it as `../data/amazon_reviews_cleaned.csv`.

### 4. Run notebooks in order

Open Jupyter and run the notebooks sequentially from 01 through 11. Each notebook saves its output to the `data/` folder for the next notebook to load.

```bash
jupyter notebook
```

### 5. Launch the Streamlit application

After running Notebook 07 to generate the model artefacts:

```bash
python -m streamlit run app.py
```

Or visit the live deployment at:

**[https://customer-feedback-sentiment-analyser.streamlit.app](https://customer-feedback-sentiment-analyser.streamlit.app)**

---

## Key Results

| Model | Weighted F1 | ROC-AUC | Notes |
|-------|------------|---------|-------|
| Complement Naive Bayes | ~0.79 | ~0.88 | Fast baseline, handles imbalance well |
| Logistic Regression (balanced) | ~0.87 | ~0.93 | Best classical model |
| Random Forest (balanced) | ~0.84 | ~0.91 | Strong but slower to train |
| XGBoost | ~0.85 | ~0.92 | Competitive with Logistic Regression |
| Bi-LSTM | ~0.83 | ~0.90 | Deep learning sequential model |
| DistilBERT (fine-tuned) | ~0.89 | ~0.95 | Best overall, requires GPU |

> Performance estimates based on the 80/20 stratified test split. All models trained with explicit class imbalance handling. The majority class baseline (always predict Negative) achieves 68.2% accuracy, all models substantially exceed this.

**Recommended production architecture:** Logistic Regression as the primary high-throughput classifier, with DistilBERT as a second-opinion layer for predictions below 65% confidence.

---

## Key Business Findings

- **Account and Billing** issues drive a 92% negative rate which is the single highest priority operational problem
- **Returns and Refunds** follow at 90% negative rate requiring immediate process improvement
- **Customer Service** is mentioned 5,998 times in negative reviews which is the most cited complaint theme
- Delivery failures and account management issues are the two most frequently discussed complaint themes
- Home and Living and Sports categories show the highest negative review rates at 70.0% and 67.7%
- The United States and United Kingdom together account for approximately 80% of reviews
- Canada (77.7%) and Ireland (74.8%) show the highest negative rates among individual markets
- Review volume increased sharply from 2019 onward with 2023 being the peak year at 4,017 reviews
- Fast delivery and product quality are the primary drivers of positive sentiment

---

## Business Recommendations

| Priority | Recommendation | Expected Impact |
|----------|---------------|----------------|
| CRITICAL | Fix account management - audit suspension algorithm, add 24hr reinstatement SLA | Reduce account topic negative rate from 92% to below 75% |
| HIGH | Accelerate refunds - auto-approve under £50, extend return window to 60 days | Reduce returns topic negative rate from 90% to below 70% |
| HIGH | Overhaul customer service - 4hr first response SLA, sentiment-based triage | Reduce CS mentions from 5,998 to below 3,000 |
| MEDIUM | Strengthen delivery reliability - second logistics partner for Q4 | Reduce delivery topic negative rate from 85% |
| MEDIUM | Deploy automated sentiment pipeline - weekly digest, real-time triage | Process 20,000+ reviews automatically, saving 400+ analyst hours per month |

**6-Month Target:** Reduce overall negative review rate from 68.2% to below 60%

---

## Project Structure Notes

- All notebooks are self-contained with their own install and data loading cells and can be run independently
- The `src/` module extracts the reusable functions from the notebooks for clean production use
- The `tests/` folder provides 35 unit tests for the preprocessing and inference functions - all passing
- Model artefacts are excluded from version control via `.gitignore` because of file size
- The dataset switch from synthetic to real Amazon data is documented in full in the section above

---

## Dependencies

See `requirements.txt` for the full list. Core dependencies:

- pandas, numpy
- matplotlib, seaborn, plotly
- nltk, textblob, langdetect
- scikit-learn, xgboost
- transformers, torch
- wordcloud
- streamlit

---

## Live Application

The deployment application is hosted on Streamlit Cloud and available at:

**[https://customer-feedback-sentiment-analyser.streamlit.app]**

The app provides four pages:

| Page | Description |
|------|-------------|
| Single Review | Paste any customer review for instant sentiment classification with confidence score |
| Batch Analysis | Upload a CSV of reviews and download predictions for all of them |
| Dashboard Summary | View key findings, model rankings, and business recommendations |
| About | Model pipeline description and deployment stack |

---

## Author

**Ifeoluwa Adebiyi**
