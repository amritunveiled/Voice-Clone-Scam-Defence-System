from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Iterable

import numpy as np
from rapidfuzz.fuzz import ratio
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

LABELS = ("benign", "urgent_legit", "scam")

# Detection-oriented terms. They are intentionally generic indicators, not
# operational scam instructions.
LEXICON = {
    "english": {
        "urgent_legit": [
            "urgent", "quickly", "emergency", "time sensitive", "call back", "hospital",
        ],
        "scam": [
            "otp", "password", "transfer", "money", "payment", "account", "private account",
            "send money", "transfer funds", "immediate transfer", "unexpected call",
        ],
    },
    "hindi": {
        "urgent_legit": [
            "जरूरी", "जल्दी", "आपात", "तुरंत जवाब", "अस्पताल", "परिवार से पुष्टि",
            "वापस फोन", "जल्द से जल्द",
            "zaroori", "jaldi", "aapat", "aspatal", "wapas phone", "jaldi se jaldi",
        ],
        "scam": [
            "ओटीपी", "पासवर्ड", "पैसे", "पैसा", "भेजो", "ट्रांसफर", "भुगतान", "खाते की जानकारी",
            "तुरंत पैसे", "अचानक फोन", "otp", "password", "paisa", "paise", "transfer", "payment",
            "account", "khata", "paise bhejo",
        ],
    },
    "kannada": {
        "urgent_legit": [
            "ತುರ್ತು", "ಬೇಗ", "ಆಸ್ಪತ್ರೆ", "ಪರಿಶೀಲಿಸಿ", "ಮತ್ತೆ ಕರೆ", "ಶೀಘ್ರವಾಗಿ",
            "turthu", "bega", "aspatre", "parishilisi", "matte kare", "sheeghravagi",
        ],
        "scam": [
            "ಒಟಿಪಿ", "ಪಾಸ್ವರ್ಡ್", "ಹಣ", "ವರ್ಗಾಯಿಸಿ", "ಪಾವತಿ", "ಖಾತೆ", "ಅನಿರೀಕ್ಷಿತ ಕರೆ",
            "ತಕ್ಷಣ ಹಣ", "otp", "password", "hana", "vargayisi", "payment", "khate",
            "anireekshita kare",
        ],
    },
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def language_group(language: str) -> str:
    x = normalize_text(language).replace("_", "-")
    if x in {"en", "english"}:
        return "english"
    if x in {"hi", "hindi"}:
        return "hindi"
    if x in {"kn", "kannada"}:
        return "kannada"
    return "english"


def fuzzy_lexicon_scores(text: str, language: str) -> dict[str, float]:
    """Return max fuzzy indicator per class, normalized to [0,1]."""
    t = normalize_text(text)
    lang = language_group(language)
    scores = {"benign": 0.0, "urgent_legit": 0.0, "scam": 0.0}

    for label in ("urgent_legit", "scam"):
        terms = LEXICON.get(lang, {}).get(label, [])
        for term in terms:
            term_n = normalize_text(term)
            if not term_n:
                continue
            if term_n in t:
                score = 1.0
            else:
                # Fuzzy comparison against words and short windows prevents
                # minor ASR spelling errors from eliminating the signal.
                tokens = t.split()
                candidates = tokens[:]
                for i in range(len(tokens) - 1):
                    candidates.append(tokens[i] + " " + tokens[i + 1])
                score = max((ratio(term_n, c) / 100.0 for c in candidates), default=0.0)
            scores[label] = max(scores[label], score)

    # Benign gets the residual only when no risk lexicon is present.
    scores["benign"] = max(0.0, 1.0 - max(scores["urgent_legit"], scores["scam"]))
    return scores


def build_fuzzy_matrix(texts: Iterable[str], languages: Iterable[str]) -> np.ndarray:
    rows = []
    for text, lang in zip(texts, languages):
        s = fuzzy_lexicon_scores(text, lang)
        # Columns: benign, urgent_legit, scam
        rows.append([s["benign"], s["urgent_legit"], s["scam"]])
    return np.asarray(rows, dtype=np.float32)


@dataclass
class IntentModel:
    vectorizer: TfidfVectorizer
    classifier: LogisticRegression
    fuzzy_weight: float = 0.35

    def _char_probs(self, texts: list[str]) -> np.ndarray:
        x = self.vectorizer.transform([normalize_text(t) for t in texts])
        return self.classifier.predict_proba(x)

    def predict_proba(self, texts: list[str], languages: list[str]) -> np.ndarray:
        char_probs = self._char_probs(texts)
        fuzzy = build_fuzzy_matrix(texts, languages)
        fuzzy_prior = fuzzy / np.maximum(fuzzy.sum(axis=1, keepdims=True), 1e-8)
        mixed = (1.0 - self.fuzzy_weight) * char_probs + self.fuzzy_weight * fuzzy_prior
        mixed /= np.maximum(mixed.sum(axis=1, keepdims=True), 1e-8)
        return mixed

    def predict(self, texts: list[str], languages: list[str]) -> np.ndarray:
        probs = self.predict_proba(texts, languages)
        return np.asarray([LABELS[i] for i in probs.argmax(axis=1)])


def fit_intent_model(texts: list[str], labels: list[str]) -> IntentModel:
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
        lowercase=True,
        max_features=20000,
    )
    x = vectorizer.fit_transform([normalize_text(t) for t in texts])
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=12345,
    )
    clf.fit(x, labels)
    return IntentModel(vectorizer=vectorizer, classifier=clf)
