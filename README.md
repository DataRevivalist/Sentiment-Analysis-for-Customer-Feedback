# ShopEase Europe - Sentiment Analysis for Customer Feedback
> **Automated sentiment classification and insight generation from real Amazon customer reviews**.

\---

## Project Overview

ShopEase Europe is a fast-growing e-commerce company with operations across the United Kingdom, Germany, France, and Spain. The company receives large volumes of customer feedback daily and needed an automated solution to classify sentiment, surface recurring complaint themes, and generate actionable business insights at scale.

This project delivers a complete end-to-end machine learning pipeline that:

* Classifies customer reviews as **Positive**, **Neutral**, or **Negative**
* Identifies key discussion themes using topic modelling
* Surfaces the primary drivers of customer satisfaction and dissatisfaction
* Provides an interactive dashboard for business stakeholders
* Includes a deployment-ready Streamlit application for real-time prediction

\---

## Dataset

|Property|Value|
|-|-|
|Source|Amazon customer reviews (real data)|
|File|`data/amazon\\\_reviews\\\_cleaned.csv`|
|Total reviews|21,055|
|After cleaning|20,407|
|Date range|August 2007 to September 2024|
|Countries|148 unique countries|
|Product categories|Sports, Electronics, Fashion, Beauty, Toys, Home and Living, Food and Grocery|
|Sentiment distribution|68.2% Negative, 27.6% Positive, 4.2% Neutral|
|Language|Predominantly English|

The dataset has a strong negative-majority class distribution, which is consistent with real-world Amazon review behaviour where dissatisfied customers are significantly more motivated to leave feedback than satisfied ones.

\---

## Repository Structure

```
shopease-sentiment-analysis/
│
├── data/
│   ├── amazon\\\_reviews\\\_cleaned.csv          # Raw input dataset
│   ├── amazon\\\_reviews\\\_cleaned\\\_processed.csv # After cleaning (Notebook 2 output)
│   ├── reviews\\\_with\\\_lang.csv               # After language detection (Notebook 3 output)
│   ├── reviews\\\_preprocessed.csv            # After NLP preprocessing (Notebook 4 output)
│   └── reviews\\\_with\\\_topics.csv             # After topic modelling (Notebook 6 output)
│
├── notebooks/
│   ├── 01\\\_data\\\_audit.ipynb                 # Data structure, quality, and completeness
│   ├── 02\\\_data\\\_cleaning.ipynb              # Deduplication, null handling, feature derivation
│   ├── 03\\\_language\\\_detection.ipynb         # Automated language identification per review
│   ├── 04\\\_text\\\_preprocessing.ipynb         # NLP pipeline and feature engineering
│   ├── 05\\\_eda.ipynb                        # Exploratory analysis across sentiment, geography, time
│   ├── 06\\\_topic\\\_modelling.ipynb            # LDA topic discovery and labelling
│   ├── 07\\\_classical\\\_modelling.ipynb        # Naive Bayes, Logistic Regression, Random Forest, XGBoost
│   └── 08\\\_transformer\\\_modelling.ipynb      # DistilBERT fine-tuning and zero-shot classification
│
├── models/
│   ├── tfidf\\\_vectoriser.pkl                # Fitted TF-IDF vectoriser (Notebook 7 output)
│   ├── label\\\_encoder.pkl                   # Label encoder for sentiment classes
│   ├── best\\\_model\\\_Logistic\\\_Regression.pkl  # Best classical model (Notebook 7 output)
│   └── distilbert/                         # Fine-tuned DistilBERT model files
│       ├── config.json
│       ├── pytorch\\\_model.bin
│       └── tokenizer files
│
├── reports/
│   ├── figures/                            # All charts and visualisations saved from notebooks
│   │   ├── fig\\\_data\\\_audit.png
│   │   ├── fig\\\_cleaning\\\_overview.png
│   │   ├── fig\\\_language.png
│   │   ├── fig\\\_vader\\\_validation.png
│   │   ├── fig\\\_eda\\\_sentiment\\\_rating.png
│   │   ├── fig\\\_eda\\\_country.png
│   │   ├── fig\\\_eda\\\_trend.png
│   │   ├── fig\\\_category\\\_insights.png
│   │   ├── fig\\\_wordclouds.png
│   │   ├── fig\\\_ngrams.png
│   │   ├── fig\\\_complaints\\\_praise.png
│   │   ├── fig\\\_topics.png
│   │   ├── fig\\\_confusion\\\_matrices.png
│   │   ├── fig\\\_model\\\_comparison.png
│   │   ├── fig\\\_feature\\\_importance.png
│   │   └── fig\\\_distilbert\\\_confusion.png
│   └── ShopEase\\\_Methodology\\\_Justification.docx
│
├── src/
│   ├── \\\_\\\_init\\\_\\\_.py
│   ├── preprocessing.py                    # Reusable text cleaning functions
│   ├── predict.py                          # Inference functions for loading and running models
│   └── app.py                              # Streamlit deployment application
│
├── tests/
│   ├── \\\_\\\_init\\\_\\\_.py
│   ├── test\\\_preprocessing.py               # Unit tests for text cleaning pipeline
│   └── test\\\_predict.py                     # Unit tests for inference functions
│
├── .gitignore
├── requirements.txt
└── README.md
```

\---

## How to Run

### 1\. Clone the repository

```bash
git clone https://github.com/your-username/shopease-sentiment-analysis.git
cd shopease-sentiment-analysis
```

### 2\. Install dependencies

```bash
pip install -r requirements.txt
```

### 3\. Set up the data directory

Place `amazon\\\_reviews\\\_cleaned.csv` inside the `data/` folder. The notebooks reference it as `../data/amazon\\\_reviews\\\_cleaned.csv`.

### 4\. Run notebooks in order

Open Jupyter and run the notebooks sequentially from 01 through 08. Each notebook saves its output to the `data/` folder for the next notebook to load.

```bash
jupyter notebook
```

### 5\. Launch the Streamlit application

After running Notebook 07 to generate the model artefacts:

```bash
streamlit run src/app.py
```

\---

## Key Results

|Model|Weighted F1|ROC-AUC|Notes|
|-|-|-|-|
|Complement Naive Bayes|\~0.79|\~0.88|Fast baseline, handles imbalance well|
|Logistic Regression (balanced)|\~0.87|\~0.93|Best classical model|
|Random Forest (balanced)|\~0.84|\~0.91|Strong but slower to train|
|XGBoost|\~0.85|\~0.92|Competitive with LR|
|DistilBERT (fine-tuned)|\~0.89|\~0.95|Best overall, requires GPU|

> Performance estimates based on the 80/20 stratified test split. Actual values depend on random seed and hardware.

\---

## Key Business Findings

* Delivery failures and account management issues are the two most frequently discussed complaint themes
* Home and Living and Sports categories show the highest negative review rates
* The United States and United Kingdom together account for approximately 80% of reviews
* Review volume increased sharply from 2019 onward, with 2023 being the peak year
* Fast delivery and product quality are the primary drivers of positive sentiment

\---

## Project Structure Notes

* All notebooks are self-contained with their own install and data loading cells and can be run independently
* The `src/` module extracts the reusable functions from the notebooks for clean production use
* The `tests/` folder provides unit tests for the preprocessing and inference functions
* Model artefacts are excluded from version control via `.gitignore` because of file size

\---

## Dependencies

See `requirements.txt` for the full list. Core dependencies:

* pandas, numpy
* matplotlib, seaborn, plotly
* nltk, textblob, langdetect
* scikit-learn, xgboost
* transformers, torch
* wordcloud
* streamlit

\---

## Author

Ifeoluwa Adebiyi
June 2026

