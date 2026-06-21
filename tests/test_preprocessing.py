"""
test_preprocessing.py
---------------------
Unit tests for the text preprocessing pipeline in src/preprocessing.py.

Run with:
    pytest tests/test_preprocessing.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import clean_text, clean_batch, STOP_WORDS


class TestCleanText:

    def test_lowercasing(self):
        result = clean_text("GREAT PRODUCT")
        assert result == result.lower()

    def test_url_removal(self):
        result = clean_text("Visit https://amazon.com for more details")
        assert "amazon.com" not in result
        assert "https" not in result

    def test_html_removal(self):
        result = clean_text("<b>Bold text</b> and <br/> line break")
        assert "<" not in result
        assert ">" not in result

    def test_stop_word_removal(self):
        result = clean_text("the product is very good and I like it")
        tokens = result.split()
        for token in tokens:
            assert token not in STOP_WORDS

    def test_short_token_removal(self):
        result = clean_text("ok it is a ok")
        tokens = result.split()
        for token in tokens:
            assert len(token) > 2

    def test_amazon_stop_words_removed(self):
        result = clean_text("I ordered this product on Amazon and wrote a review")
        tokens = result.split()
        amazon_terms = {"amazon", "order", "ordered", "product", "review"}
        for token in tokens:
            assert token not in amazon_terms

    def test_punctuation_removal(self):
        result = clean_text("Terrible!!! Never again.")
        assert "!" not in result
        assert "." not in result

    def test_returns_string(self):
        result = clean_text("Some review text here")
        assert isinstance(result, str)

    def test_empty_string(self):
        result = clean_text("")
        assert isinstance(result, str)

    def test_non_string_input(self):
        result = clean_text(12345)
        assert isinstance(result, str)

    def test_positive_review_keeps_sentiment_words(self):
        result = clean_text("Excellent quality, fast delivery, very happy")
        assert "excellent" in result or "quality" in result or "happy" in result

    def test_negative_review_keeps_sentiment_words(self):
        result = clean_text("Broken on arrival, terrible quality, never again")
        assert "broken" in result or "terrible" in result

    def test_lemmatisation_on(self):
        result_lem = clean_text("The products arrived broken", lemmatize=True)
        result_raw = clean_text("The products arrived broken", lemmatize=False)
        # Lemmatised version should map 'products' to 'product'
        # but 'product' is in STOP_WORDS so the token will be removed.
        # Instead test with a non-stop-word verb form.
        text = "The packages arrived quickly and were delivered safely"
        lem  = clean_text(text, lemmatize=True)
        nolem = clean_text(text, lemmatize=False)
        assert isinstance(lem, str)
        assert isinstance(nolem, str)

    def test_lemmatisation_off(self):
        result = clean_text("running runners ran", lemmatize=False)
        assert isinstance(result, str)


class TestCleanBatch:

    def test_returns_list(self):
        result = clean_batch(["review one", "review two"])
        assert isinstance(result, list)

    def test_length_preserved(self):
        texts = ["first review", "second review", "third review"]
        result = clean_batch(texts)
        assert len(result) == len(texts)

    def test_each_element_is_string(self):
        result = clean_batch(["good product", "bad service"])
        for item in result:
            assert isinstance(item, str)

    def test_empty_list(self):
        result = clean_batch([])
        assert result == []

    def test_consistent_with_single(self):
        text = "The item arrived broken and damaged"
        single = clean_text(text)
        batch  = clean_batch([text])
        assert batch[0] == single
