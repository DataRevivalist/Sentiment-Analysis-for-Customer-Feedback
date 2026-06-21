"""
preprocessing.py
----------------
Reusable text cleaning and preprocessing functions for the ShopEase
sentiment analysis pipeline.

These functions are extracted from Notebook 04 so they can be imported
by the inference module, the Streamlit app, and the test suite without
duplicating code.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data on first import
for _pkg in ["punkt", "stopwords", "wordnet", "punkt_tab", "omw-1.4"]:
    nltk.download(_pkg, quiet=True)

# Initialise lemmatizer once at module level for efficiency
_lemmatizer = WordNetLemmatizer()

# Standard English stop words plus Amazon-specific terms that appear
# at high frequency without adding sentiment signal
STOP_WORDS = set(stopwords.words("english")) | {
    "product", "item", "ordered", "order", "amazon", "purchase",
    "bought", "buy", "would", "also", "one", "get", "got", "use",
    "used", "using", "review", "star", "stars", "rating",
}

# Unicode emoji pattern
_EMOJI_RE = re.compile(
    "["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F9FF"
    u"\u2600-\u26FF"
    u"\u2700-\u27BF"
    "]+",
    flags=re.UNICODE,
)


def clean_text(text: str, lemmatize: bool = True) -> str:
    """
    Apply the full NLP preprocessing pipeline to a single review string.

    Steps applied in order:
        1. Lowercase
        2. Remove URLs
        3. Remove HTML tags
        4. Remove emojis
        5. Remove non-alphabetic characters
        6. Collapse whitespace
        7. Tokenise
        8. Remove stop words and tokens shorter than 3 characters
        9. Lemmatise (optional)

    Parameters
    ----------
    text : str
        Raw review text.
    lemmatize : bool, default True
        Whether to apply WordNet lemmatisation.

    Returns
    -------
    str
        Cleaned, space-joined token string ready for TF-IDF vectorisation.

    Examples
    --------
    >>> clean_text("The product arrived BROKEN!!! Terrible quality.")
    'arrived broken terrible quality'
    """
    text = str(text).lower()
    text = re.sub(r"https?\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = _EMOJI_RE.sub(" ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    if lemmatize:
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def clean_batch(texts, lemmatize: bool = True) -> list:
    """
    Apply clean_text to a list or pandas Series of review strings.

    Parameters
    ----------
    texts : list or pandas.Series
        Collection of raw review strings.
    lemmatize : bool, default True
        Whether to apply lemmatisation.

    Returns
    -------
    list of str
        Cleaned text strings in the same order as the input.
    """
    return [clean_text(t, lemmatize=lemmatize) for t in texts]
