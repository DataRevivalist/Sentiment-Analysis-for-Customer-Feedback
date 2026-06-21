"""
test_predict.py
---------------
Unit tests for the inference functions in src/predict.py.

These tests use mock objects so they can be run without real model artefacts.
To run integration tests against real models, first run Notebook 07.

Run with:
    pytest tests/test_predict.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.predict import predict_single, predict_batch


# ---------------------------------------------------------------------------
# Mock model fixtures
# ---------------------------------------------------------------------------

def make_mock_artefacts(
    predicted_class=0,
    class_probs=None,
    classes=None,
):
    """
    Build mock vectoriser, classifier, and label encoder for testing.
    """
    if classes is None:
        classes = ["Negative", "Neutral", "Positive"]
    if class_probs is None:
        class_probs = [0.75, 0.10, 0.15]

    vectoriser = MagicMock()
    vectoriser.transform.return_value = MagicMock()

    classifier = MagicMock()
    classifier.predict.return_value = np.array([predicted_class])
    classifier.predict_proba.return_value = np.array([class_probs])

    label_encoder = MagicMock()
    label_encoder.classes_ = np.array(classes)
    label_encoder.inverse_transform.return_value = np.array([classes[predicted_class]])

    return vectoriser, classifier, label_encoder


# ---------------------------------------------------------------------------
# predict_single tests
# ---------------------------------------------------------------------------

class TestPredictSingle:

    def test_returns_dict(self):
        v, c, le = make_mock_artefacts()
        result = predict_single("Some review text", v, c, le)
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        v, c, le = make_mock_artefacts()
        result = predict_single("Some review text", v, c, le)
        assert "label" in result
        assert "confidence" in result
        assert "probabilities" in result

    def test_label_is_string(self):
        v, c, le = make_mock_artefacts()
        result = predict_single("Review text", v, c, le)
        assert isinstance(result["label"], str)

    def test_label_is_valid_class(self):
        classes = ["Negative", "Neutral", "Positive"]
        v, c, le = make_mock_artefacts(classes=classes)
        result = predict_single("Review text", v, c, le)
        assert result["label"] in classes

    def test_confidence_is_float(self):
        v, c, le = make_mock_artefacts()
        result = predict_single("Review text", v, c, le)
        assert isinstance(result["confidence"], float)

    def test_confidence_between_zero_and_one(self):
        v, c, le = make_mock_artefacts(class_probs=[0.75, 0.10, 0.15])
        result = predict_single("Review text", v, c, le)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_probabilities_is_dict(self):
        v, c, le = make_mock_artefacts()
        result = predict_single("Review text", v, c, le)
        assert isinstance(result["probabilities"], dict)

    def test_probabilities_has_all_classes(self):
        classes = ["Negative", "Neutral", "Positive"]
        v, c, le = make_mock_artefacts(classes=classes)
        result = predict_single("Review text", v, c, le)
        for cls in classes:
            assert cls in result["probabilities"]

    def test_probabilities_sum_to_one(self):
        v, c, le = make_mock_artefacts(class_probs=[0.75, 0.10, 0.15])
        result = predict_single("Review text", v, c, le)
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 1e-6

    def test_negative_prediction(self):
        v, c, le = make_mock_artefacts(
            predicted_class=0,
            class_probs=[0.90, 0.05, 0.05],
            classes=["Negative", "Neutral", "Positive"]
        )
        result = predict_single("Broken and terrible", v, c, le)
        assert result["label"] == "Negative"
        assert result["confidence"] == pytest.approx(0.90)

    def test_positive_prediction(self):
        v, c, le = make_mock_artefacts(
            predicted_class=2,
            class_probs=[0.05, 0.05, 0.90],
            classes=["Negative", "Neutral", "Positive"]
        )
        result = predict_single("Excellent quality", v, c, le)
        assert result["label"] == "Positive"

    def test_handles_empty_string(self):
        v, c, le = make_mock_artefacts()
        result = predict_single("", v, c, le)
        assert isinstance(result, dict)
        assert "label" in result


# ---------------------------------------------------------------------------
# predict_batch tests
# ---------------------------------------------------------------------------

class TestPredictBatch:

    def test_returns_list(self):
        v, c, le = make_mock_artefacts()
        v.transform.return_value = MagicMock()
        le.inverse_transform.return_value = np.array(["Negative", "Positive"])
        c.predict.return_value = np.array([0, 2])
        c.predict_proba.return_value = np.array([
            [0.80, 0.10, 0.10],
            [0.10, 0.10, 0.80],
        ])
        result = predict_batch(["review one", "review two"], v, c, le)
        assert isinstance(result, list)

    def test_length_matches_input(self):
        v, c, le = make_mock_artefacts()
        n = 5
        c.predict.return_value = np.zeros(n, dtype=int)
        c.predict_proba.return_value = np.tile([0.7, 0.2, 0.1], (n, 1))
        le.inverse_transform.return_value = np.array(["Negative"] * n)
        result = predict_batch(["text"] * n, v, c, le)
        assert len(result) == n

    def test_each_element_has_required_keys(self):
        v, c, le = make_mock_artefacts()
        n = 3
        c.predict.return_value = np.zeros(n, dtype=int)
        c.predict_proba.return_value = np.tile([0.7, 0.2, 0.1], (n, 1))
        le.inverse_transform.return_value = np.array(["Negative"] * n)
        results = predict_batch(["text"] * n, v, c, le)
        for r in results:
            assert "label" in r
            assert "confidence" in r
            assert "probabilities" in r

    def test_empty_input_returns_empty_list(self):
        v, c, le = make_mock_artefacts()
        c.predict.return_value = np.array([])
        c.predict_proba.return_value = np.empty((0, 3))
        le.inverse_transform.return_value = np.array([])
        result = predict_batch([], v, c, le)
        assert result == []
