import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)

STOP = set(stopwords.words("english"))
STEM = PorterStemmer()


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"^subject:", " ", text.strip())
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [STEM.stem(w) for w in text.split() if w not in STOP and len(w) > 2]
    return " ".join(words)
