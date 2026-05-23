import re
from functools import lru_cache

import nltk
import pymorphy3
from nltk.stem import WordNetLemmatizer


# Токен — последовательность русских/английских букв, цифр или нижних подчёркиваний.
TOKEN_RE = re.compile(r"[a-zа-яё0-9_]+", re.IGNORECASE)
RU_RE = re.compile(r"^[а-яё]+$")
EN_RE = re.compile(r"^[a-z]+$")

morph = pymorphy3.MorphAnalyzer()
eng_lemmatizer = WordNetLemmatizer()


def ensure_nltk_data() -> None:
    """
    WordNetLemmatizer требует словарь wordnet.
    """
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)

    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4", quiet=True)


@lru_cache(maxsize=200_000)
def lemmatize_token(token: str) -> str:
    """
    Приводит один токен к начальной форме:
    - русские слова обрабатывает pymorphy3;
    - английские слова обрабатывает WordNetLemmatizer;
    - смешанные токены, числа и токены с цифрами возвращает как есть.
    """
    token = token.lower()

    if RU_RE.match(token):
        return morph.parse(token)[0].normal_form

    if EN_RE.match(token):
        return eng_lemmatizer.lemmatize(token)

    return token


def preprocess_text(text) -> list[str]:
    """
    Предобработка текста для индекса:
    1. Проверяем, что вход — строка.
    2. Приводим текст к нижнему регистру.
    3. Достаём токены регулярным выражением.
    4. Лемматизируем каждый токен.

    Важно:
    - стоп-слова НЕ удаляются;
    - короткие слова НЕ удаляются.
    """
    if not isinstance(text, str):
        return []

    text = text.lower()
    raw_tokens = TOKEN_RE.findall(text)

    return [lemmatize_token(token) for token in raw_tokens]
