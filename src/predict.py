"""
predict.py
----------
Inference functions for loading the trained model artefacts and running
sentiment predictions on new review text.

These functions are used by the Streamlit app (src/app.py) and can be
imported directly for batch processing in production pipelines.
"""

import pickle
import glob
import os
import numpy as np
from typing import Union

from src.preprocessing import clean_text, clean_batch


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_artefacts(models_dir: str = "../models"):
    """
    Load the TF-IDF vectoriser, label encoder, and best classical model
    from the models directory.

    Parameters
    ----------
    models_dir : str
        Path to the directory containing the saved .pkl artefacts.

    Returns
    -------
    tuple of (vectoriser, classifier, label_encoder)

    Raises
    ------
    FileNotFoundError
        If any of the required artefact files are missing.
    """
    vectoriser_path = os.path.join(models_dir, "tfidf_vectoriser.pkl")
    encoder_path    = os.path.join(models_dir, "label_encoder.pkl")
    model_paths     = glob.glob(os.path.join(models_dir, "best_model_*.pkl"))

    if not os.path.exists(vectoriser_path):
        raise FileNotFoundError(
            f"TF-IDF vectoriser not found at {vectoriser_path}. "
            "Run Notebook 07 first to generate model artefacts."
        )
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(
            f"Label encoder not found at {encoder_path}."
        )
    if not model_paths:
        raise FileNotFoundError(
            f"No best_model_*.pkl found in {models_dir}. "
            "Run Notebook 07 first."
        )

    with open(vectoriser_path, "rb") as f:
        vectoriser = pickle.load(f)
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    with open(model_paths[0], "rb") as f:
        classifier = pickle.load(f)

    return vectoriser, classifier, label_encoder


# ---------------------------------------------------------------------------
# Single review prediction
# ---------------------------------------------------------------------------

def predict_single(
    text: str,
    vectoriser,
    classifier,
    label_encoder,
) -> dict:
    """
    Predict the sentiment of a single review string.

    Parameters
    ----------
    text : str
        Raw review text.
    vectoriser : sklearn TfidfVectorizer
        Fitted vectoriser loaded from models/.
    classifier : sklearn estimator
        Fitted classifier loaded from models/.
    label_encoder : sklearn LabelEncoder
        Fitted label encoder loaded from models/.

    Returns
    -------
    dict with keys:
        - 'label'       : str, predicted sentiment class
        - 'confidence'  : float, probability of the predicted class
        - 'probabilities': dict mapping each class name to its probability
    """
    cleaned = clean_text(text)
    X = vectoriser.transform([cleaned])
    pred_idx  = classifier.predict(X)[0]
    pred_prob = classifier.predict_proba(X)[0]

    label = label_encoder.inverse_transform([pred_idx])[0]
    classes = label_encoder.classes_

    return {
        "label":          label,
        "confidence":     float(np.max(pred_prob)),
        "probabilities":  {cls: float(prob) for cls, prob in zip(classes, pred_prob)},
    }


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

def predict_batch(
    texts,
    vectoriser,
    classifier,
    label_encoder,
) -> list:
    """
    Predict sentiment for a list of review strings.

    Parameters
    ----------
    texts : list of str or pandas.Series
        Raw review texts.
    vectoriser, classifier, label_encoder
        Artefacts returned by load_artefacts().

    Returns
    -------
    list of dict
        One result dict per input text, in the same order.
        Each dict has the same keys as predict_single().
    """
    cleaned = clean_batch(list(texts))
    X = vectoriser.transform(cleaned)
    pred_indices = classifier.predict(X)
    pred_probs   = classifier.predict_proba(X)
    classes      = label_encoder.classes_

    results = []
    for idx, probs in zip(pred_indices, pred_probs):
        label = label_encoder.inverse_transform([idx])[0]
        results.append({
            "label":         label,
            "confidence":    float(np.max(probs)),
            "probabilities": {cls: float(p) for cls, p in zip(classes, probs)},
        })
    return results
