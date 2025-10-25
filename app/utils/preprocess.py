import re

def preprocess_text(text: str) -> str:
    """
    Минимальная очистка текста:
    - нижний регистр
    - убираем спецсимволы, emoji, пунктуацию
    - убираем лишние пробелы
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
