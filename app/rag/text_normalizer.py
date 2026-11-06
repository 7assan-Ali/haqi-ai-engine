
import re
import unicodedata


# Arabic diacritics (Tashkeel) and Quranic annotation marks.
ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)

# Remove tatweel (ـ).
TATWEEL = "\u0640"

# Normalize common Arabic letter variations.
ARABIC_CHAR_MAP = str.maketrans({
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ٱ": "ا",
    "ى": "ي",
    "ی": "ي",  # Persian Yeh
    "ک": "ك",  # Persian Kaf
})


def normalize_arabic_text(text: str) -> str:
    """
    Normalize Arabic text for retrieval and search.

    The original input must always be preserved separately.
    This function returns a normalized copy.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    # Normalize Unicode compatibility characters.
    text = unicodedata.normalize("NFKC", text)

    # Remove Arabic diacritics.
    text = ARABIC_DIACRITICS.sub("", text)

    # Remove tatweel.
    text = text.replace(TATWEEL, "")

    # Normalize Arabic character variations.
    text = text.translate(ARABIC_CHAR_MAP)

    # Normalize whitespace without removing punctuation.
    text = re.sub(r"\s+", " ", text)

    return text.strip()
