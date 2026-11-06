
import pytest

from app.rag.text_normalizer import normalize_arabic_text


def test_normalize_alef_variations():
    result = normalize_arabic_text("أحمد، إكرام، آمال")
    assert result == "احمد، اكرام، امال"


def test_remove_diacritics():
    result = normalize_arabic_text("القَانُونُ")
    assert result == "القانون"


def test_remove_tatweel():
    result = normalize_arabic_text("القــــانون")
    assert result == "القانون"


def test_normalize_alef_maqsura():
    result = normalize_arabic_text("على الهدى")
    assert result == "علي الهدي"


def test_normalize_persian_characters():
    result = normalize_arabic_text("کلیه ی")
    assert result == "كليه ي"


def test_normalize_whitespace():
    result = normalize_arabic_text("  قانون   رقم\n  ١٠ ")
    assert result == "قانون رقم ١٠"


def test_preserve_punctuation_and_ta_marbuta():
    result = normalize_arabic_text("المادة (١)؛ الخاصة بالملكية.")
    assert result == "المادة (١)؛ الخاصة بالملكية."


def test_empty_string():
    assert normalize_arabic_text("") == ""


def test_non_string_input():
    with pytest.raises(TypeError):
        normalize_arabic_text(None)


def test_original_text_is_not_modified():
    original = "إِنَّ القانونَ"

    normalized = normalize_arabic_text(original)

    assert original == "إِنَّ القانونَ"
    assert normalized == "ان القانون"
